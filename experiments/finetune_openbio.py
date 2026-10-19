"""
finetune_openbio.py
-------------------
LoRA fine-tuning of aaditya/Llama3-OpenBioLLM-8B on a mental-health JSONL
dataset, followed by ANLS evaluation on the test set. Self-contained.

OpenBioLLM-8B is built on Meta-LLaMA-3-8B and uses the identical LLaMA-3 chat
template, so the pipeline matches the 3B script; only the defaults differ
(larger model → smaller batch, grad_accum=2, lower lr).

USAGE:
  python finetune_openbio.py --train train_mh_tech_dataset.jsonl \
                             --test  test_mh_tech_dataset.jsonl

  # override any default:
  python finetune_openbio.py --train t.jsonl --test e.jsonl \
      --epochs 3 --batch 16 --lr 1e-4 --lora_r 16

  # quick smoke test on a subset:
  python finetune_openbio.py --train t.jsonl --test e.jsonl --n_train 200 --n_test 50

RESULTS:
  Saved to ./results/<train_basename>__<test_basename>/results.json
  (the trained adapter is NOT saved).

REQUIREMENTS:
  pip install transformers peft accelerate datasets
"""

import argparse, gc, json, os, random, re, unicodedata, warnings
from datetime import datetime
import functools
import sys as _sys
try:  # ensure UTF-8 console output on Windows (avoids cp1252 UnicodeEncodeError)
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
import logging; logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, TaskType

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID       = "aaditya/Llama3-OpenBioLLM-8B"
EPOCHS         = 3
BATCH_SIZE     = 32          # 8B model
GRAD_ACCUM     = 2           # effective batch = BATCH_SIZE x GRAD_ACCUM = 64
LEARNING_RATE  = 1e-4
WARMUP_RATIO   = 0.05
MAX_SEQ_LEN    = 1024       # fits long inputs (e.g. DASS 42-item) + the answer
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
EVAL_BATCH     = 16
MAX_NEW_TOKENS = 150
ANLS_THRESHOLD = 0.5

RANDOM_SEED  = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS       = torch.cuda.device_count() if torch.cuda.is_available() else 0
TRAIN_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True          # opt: autotune conv/matmul kernels
try:
    torch.set_float32_matmul_precision("high")        # opt: faster fp32 matmuls (TF32)
except Exception:
    pass

SYSTEM_PROMPT = ("You are a mental health assessment assistant. Answer the "
                 "question concisely and directly based on the given information.")
LLAMA3_TEMPLATE = (
    "{%- set loop_messages = messages -%}"
    "{%- for message in loop_messages -%}"
    "{%- set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' -%}"
    "{%- if loop.index0 == 0 -%}{%- set content = bos_token + content -%}{%- endif -%}"
    "{{ content }}{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif -%}")


# ══ DATA ══════════════════════════════════════════════════════════════════════
# Field-name aliases — different datasets use different key names; resolve to a
# common (instruction, input, output) triple so the same code works everywhere.
INSTRUCTION_KEYS = ("instruction", "question", "prompt", "query", "instructions")
INPUT_KEYS       = ("input", "context", "text", "passage", "content")
OUTPUT_KEYS      = ("output", "response", "answer", "completion", "label", "target", "gt")


def _txt(v):
    """Coerce any JSON value (str / int / float / None / list / dict) to a string,
    so .strip() never throws on a non-string field."""
    return "" if v is None else (v if isinstance(v, str) else str(v))


def _field(sample, keys):
    """Return the first present, non-empty aliased field as a string."""
    for k in keys:
        if k in sample:
            v = _txt(sample[k]).strip()
            if v:
                return v
    return ""


def _first_present(sample, keys):
    """Name of the first matching key (for the schema-detection printout)."""
    for k in keys:
        if k in sample and _txt(sample[k]).strip():
            return k
    return "(none)"


def load_jsonl(path):
    """Load a JSONL file: one JSON object per line, blank lines skipped."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def format_sample(sample, tok, for_training=True):
    user = _field(sample, INSTRUCTION_KEYS)
    inp  = _field(sample, INPUT_KEYS)
    if inp:
        user += f"\n\n{inp}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user}]
    if for_training:
        msgs.append({"role": "assistant", "content": _field(sample, OUTPUT_KEYS)})
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=not for_training)


def _tokenize_sample(sample, tok, max_len):
    """Tokenize one sample into (input_ids, labels), or return ("skip", reason).

    Key fix: the FULL sequence (prompt + answer) is what the model trains on,
    but only the PROMPT may be truncated — never the answer. Long inputs (e.g.
    the DASS 42-item symptom list) would otherwise fill max_seq_len and leave
    zero answer tokens, causing every such row to be silently dropped.
    We therefore tokenize prompt and answer separately, keep the whole answer,
    and trim the prompt from the FRONT of its body if the two together exceed
    max_len (chat-template head/tail tokens are preserved)."""
    if not _field(sample, INSTRUCTION_KEYS):
        return "skip", "missing/empty instruction"
    if not _field(sample, OUTPUT_KEYS):
        return "skip", "missing/empty output"

    full_ids   = tok(format_sample(sample, tok, True))["input_ids"]
    prompt_ids = tok(format_sample(sample, tok, False))["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return "skip", "answer empty after templating"
    answer_ids = full_ids[len(prompt_ids):]          # the response tokens (+ eot)

    # If everything fits, use as-is.
    if len(prompt_ids) + len(answer_ids) <= max_len:
        ids = prompt_ids + answer_ids
        return ids, [-100] * len(prompt_ids) + answer_ids

    # Otherwise reserve room for the (untruncated) answer and trim the prompt.
    # If the answer alone is somehow longer than max_len, truncate the answer
    # tail as a last resort so at least one example survives.
    if len(answer_ids) >= max_len:
        answer_ids = answer_ids[: max_len - 1]
        return answer_ids, list(answer_ids)
    keep_prompt = max_len - len(answer_ids)
    # keep the prompt's head (instruction/system) AND its tail (template close)
    head = keep_prompt // 2
    tail = keep_prompt - head
    trimmed_prompt = prompt_ids[:head] + prompt_ids[len(prompt_ids) - tail:]
    ids = trimmed_prompt + answer_ids
    return ids, [-100] * len(trimmed_prompt) + answer_ids


class MentalHealthDataset(Dataset):
    """Pre-tokenizes every sample once into an in-memory cache.
    Uses threads (not processes) because HF fast tokenizers can't be pickled;
    the tokenizer's C++ core releases the GIL so threads still parallelize."""
    def __init__(self, samples, tok, max_len=MAX_SEQ_LEN):
        import sys, time, collections
        n_threads = min(16, os.cpu_count() or 1)   # opt: more threads (you have many cores)
        total = len(samples)
        print(f"  Pre-tokenizing {total:,} samples using {n_threads} threads ...")
        self.ids, self.labels = [], []
        skip_reasons = collections.Counter()
        skipped = done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(_tokenize_sample, s, tok, max_len) for s in samples]
            for fut in as_completed(futures):
                done += 1
                res = fut.result()
                if isinstance(res, tuple) and len(res) == 2 and res[0] == "skip":
                    skipped += 1
                    skip_reasons[res[1]] += 1
                else:
                    self.ids.append(res[0]); self.labels.append(res[1])
                if done % 500 == 0 or done == total:
                    el  = time.time() - t0
                    pct = done / total
                    bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
                    eta = (total - done) / max(done / max(el, 1e-6), 1e-6)
                    sys.stdout.write(f"\r  [{bar}] {pct*100:5.1f}%  {done}/{total}  "
                                     f"usable={len(self.ids):,}  skipped={skipped}  "
                                     f"ETA {eta:4.0f}s")
                    sys.stdout.flush()
        print(f"\n  Tokenization complete: {len(self.ids):,} usable  "
              f"{skipped} skipped  {time.time()-t0:.1f}s")
        if skip_reasons:
            for reason, cnt in skip_reasons.most_common(10):
                print(f"    skipped {cnt:,}: {reason}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = self.ids[i]
        return {"input_ids":      torch.tensor(ids, dtype=torch.long),
                "labels":         torch.tensor(self.labels[i], dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long)}


def collate_fn(batch, pad_id):
    """Left-pad all sequences in the batch to the same length."""
    m = max(b["input_ids"].shape[0] for b in batch)
    def pad(b, key, val):
        n = m - b[key].shape[0]
        return torch.cat([torch.full((n,), val, dtype=torch.long), b[key]])
    return {"input_ids":      torch.stack([pad(b, "input_ids", pad_id) for b in batch]),
            "labels":         torch.stack([pad(b, "labels", -100)      for b in batch]),
            "attention_mask": torch.stack([pad(b, "attention_mask", 0) for b in batch])}


# ══ ANLS NORMALIZATION  (do NOT change — required for scoring) ═════════════════
def normalize_text(text):
    text = unicodedata.normalize("NFD", text.lower().strip())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

SEVERITY_RANK    = {"normal": 0, "mild": 1, "moderate": 2,
                    "severe": 3, "extremely severe": 4}
RISK_TO_SEVERITY = {"low risk": "mild", "moderate risk": "moderate",
                    "high risk": "severe", "very high risk": "extremely severe"}
SLEEP_SYNONYMS   = {"inadequate": "inadequate",
                    "severely sleep deprived": "inadequate",
                    "less than 5 hours": "inadequate", "insufficient": "insufficient",
                    "not enough": "insufficient", "5 6 hours": "insufficient",
                    "adequate": "adequate", "sufficient": "adequate",
                    "7 8 hours": "adequate", "excessive": "excessive",
                    "too much": "excessive", "more than 8 hours": "excessive"}
POSITIVE_SYNONYMS = {"yes", "yep", "yup", "ya", "affirmative", "true", "correct",
                     "sought", "received treatment", "getting help",
                     "is experiencing", "shows signs", "does have", "currently has",
                     "clinically significant", "diagnosed", "indeed", "definitely",
                     "certainly", "suffering from", "positive for", "confirms"}
NEGATIVE_SYNONYMS = {"no", "nope", "false", "incorrect", "has not", "have not",
                     "hasn't", "haven't", "does not", "doesn't", "is not", "isn't",
                     "never", "not experiencing", "no signs", "not diagnosed",
                     "within normal", "no disorder", "not present", "negative for",
                     "not detected"}


def canonical(token):
    """Map a core-answer token to canonical form (applied to BOTH pred and gt)."""
    t = token.strip().lower()
    if t in SEVERITY_RANK:    return t
    if t in RISK_TO_SEVERITY: return RISK_TO_SEVERITY[t]
    for phrase, canon in SLEEP_SYNONYMS.items():
        if phrase in t: return canon
    for syn in POSITIVE_SYNONYMS:
        if syn in t: return "yes"
    for syn in NEGATIVE_SYNONYMS:
        if syn in t: return "no"
    return t


def severity_partial_credit(p, g):
    if p == g: return 1.0
    if p in SEVERITY_RANK and g in SEVERITY_RANK \
       and abs(SEVERITY_RANK[p] - SEVERITY_RANK[g]) == 1:
        return 0.5
    return 0.0


def extract_core_answer(text):
    """Extract the short core answer. A leading Yes/No is decisive for binary
    questions and is checked FIRST, so an explanation that mentions sleep or
    severity words (e.g. 'Yes ... insufficient sleep') is not mislabeled."""
    t = text.lower().strip()
    if re.match(r"^\W*yes\b", t): return "yes"
    if re.match(r"^\W*no\b",  t): return "no"
    if re.search(r"extremely\s+severe|very\s+severe", t):     return "extremely severe"
    if re.search(r"\bsevere\b", t) and "extremely" not in t: return "severe"
    if re.search(r"\bmoderate\b", t) and "risk" not in t:    return "moderate"
    if re.search(r"\bmild\b", t):                            return "mild"
    if re.search(r"\bnormal\b", t) and "abnormal" not in t:  return "normal"
    if re.search(r"\bvery\s+high\s+risk\b", t): return "extremely severe"
    if re.search(r"\bhigh\s+risk\b", t):        return "severe"
    if re.search(r"\bmoderate\s+risk\b", t):    return "moderate"
    if re.search(r"\blow\s+risk\b", t):         return "mild"
    if re.search(r"\binadequate\b|severely\s+sleep.deprived|less\s+than\s+5", t): return "inadequate"
    if re.search(r"\binsufficient\b|not\s+enough\s+sleep|5.?6\s+hour", t):        return "insufficient"
    if re.search(r"\badequate\b|\bsufficient\b|7.?8\s+hour", t):                  return "adequate"
    if re.search(r"\bexcessive\b|too\s+much\s+sleep|more\s+than\s+8", t):         return "excessive"
    if re.search(r"\b(yes|affirmative|sought|received treatment|getting help|"
                 r"is experiencing|shows signs|does have|currently has|"
                 r"clinically significant|diagnosed|indeed|suffering from|"
                 r"positive for|confirms)\b", t): return "yes"
    if re.search(r"\bno\b|not\s|never\s|within normal|negative for", t): return "no"
    for cond in ("depression", "anxiety", "stress"):
        if re.search(rf"\b{cond}\b", t): return cond
    return normalize_text(text)[:60]


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (prev if s1[i-1] == s2[j-1]
                                  else 1 + min(prev, dp[j], dp[j-1]))
    return dp[n]


def nls(pred_core, gt_core, threshold=ANLS_THRESHOLD):
    p = canonical(normalize_text(pred_core))
    g = canonical(normalize_text(gt_core))
    if not p and not g: return 1.0
    if not p or  not g: return 0.0
    if p == g:          return 1.0
    partial = severity_partial_credit(p, g)
    if partial > 0:     return partial
    sim = 1.0 - levenshtein(p, g) / max(len(p), len(g))
    return sim if sim >= threshold else 0.0


def compute_anls(pred, gt, threshold=ANLS_THRESHOLD):
    pc, gc = extract_core_answer(pred), extract_core_answer(gt)
    return nls(pc, gc, threshold), pc, gc


# ══ MODEL ═════════════════════════════════════════════════════════════════════
def load_base_model(model_id):
    print("  Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_TEMPLATE
        print("  Chat template set manually (LLaMA-3 instruct format)")
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"; print("  Flash Attention 2 — enabled")
    except ImportError:
        attn = "sdpa";             print("  Using SDPA attention")
    print(f"  Loading model onto {TRAIN_DEVICE} (single-GPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map={"": TRAIN_DEVICE},
        trust_remote_code=True, attn_implementation=attn)
    return tok, model


def apply_lora(model):
    model.enable_input_require_grads()   # needed for grad-checkpointing + LoRA
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none"))
    model.print_trainable_parameters()
    return model


# ══ TRAIN ═════════════════════════════════════════════════════════════════════
def train(model, loader, epochs, lr, grad_accum):
    import time
    dev = next(model.parameters()).device      # use the model's real device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * epochs // grad_accum
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * WARMUP_RATIO), total_steps)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    print(f"\n  Epochs {epochs} | batches/epoch {len(loader)} | "
          f"opt steps {total_steps} | lr {lr:.2e}")
    model.train()
    model.config.use_cache = False          # opt: disable KV cache during training
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    step = 0; losses = []; t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_loss = n = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader, 1):
            ids  = batch["input_ids"].to(dev, non_blocking=True)
            mask = batch["attention_mask"].to(dev, non_blocking=True)
            lbl  = batch["labels"].to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=lbl).loss / grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            ep_loss += loss.item() * grad_accum; n += 1
            if i % grad_accum == 0:
                if scaler: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if scaler: scaler.step(opt); scaler.update()
                else:      opt.step()
                sched.step(); opt.zero_grad(set_to_none=True); step += 1
            print(f"\r  E{epoch}/{epochs} b{i}/{len(loader)}  "
                  f"loss={loss.item()*grad_accum:.4f}  avg={ep_loss/n:.4f}",
                  end="", flush=True)
        losses.append(ep_loss / max(n, 1))
        print(f"\n  Epoch {epoch} done — avg loss {losses[-1]:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)")
    print(f"\n  Training complete — {(time.time()-t0)/60:.1f} min, {step} steps, "
          f"losses {[round(l,4) for l in losses]}")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True           # re-enable for generation
    return model


# ══ EVAL ══════════════════════════════════════════════════════════════════════
def _eos_ids(tok):
    ids = [tok.eos_token_id]
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot and eot != tok.eos_token_id:
            ids.append(eot)
    except Exception:
        pass
    return ids


@torch.no_grad()
def evaluate(model, tok, samples, batch_size=EVAL_BATCH):
    model.eval()
    model.config.use_cache = True
    # NOTE: evaluate on whatever device the model is already on — do NOT move
    # it to a separate "eval GPU". Pinning eval to a fixed device was crashing.
    dev = next(model.parameters()).device
    eos = _eos_ids(tok)
    scores, results = [], []
    print(f"\n  {'#':>5}  {'Pred core':<22}{'GT core':<22}{'ANLS':>6}")
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        prompts  = [format_sample(s, tok, False) for s in batch]
        expected = [s["output"].strip() for s in batch]
        inputs = tok(prompts, return_tensors="pt", truncation=True,
                     max_length=2048, padding=True).to(dev)
        in_len = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id,
                             eos_token_id=eos, use_cache=True)
        for j, (seq, exp, s) in enumerate(zip(out, expected, batch)):
            gen = tok.decode(seq[in_len:], skip_special_tokens=True).strip()
            sc, pc, gc = compute_anls(gen, exp)
            scores.append(sc)
            results.append({"index": start + j + 1, "instruction": s["instruction"],
                            "expected": exp, "generated": gen, "pred_core": pc,
                            "gt_core": gc, "anls_score": round(sc, 4)})
            print(f"  {start+j+1:>5}  {pc[:20]:<22}{gc[:20]:<22}{sc:>6.3f}")
        print(f"  -- running ANLS @{min(start+batch_size,len(samples))}: "
              f"{sum(scores)/len(scores):.4f}")
    return sum(scores) / len(scores), results


# ══ MAIN ══════════════════════════════════════════════════════════════════════
def _result_dir(train_path, test_path):
    """Build ./results/<train_basename>__<test_basename> (extensions stripped)."""
    tr = os.path.splitext(os.path.basename(train_path))[0]
    te = os.path.splitext(os.path.basename(test_path))[0]
    d = os.path.join("results", f"{tr}__{te}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    global EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, LORA_R, LORA_ALPHA, MAX_SEQ_LEN
    p = argparse.ArgumentParser(description=f"LoRA fine-tune {MODEL_ID}")
    p.add_argument("--train", required=True)
    p.add_argument("--test",  required=True)
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--batch",      type=int,   default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int,   default=GRAD_ACCUM)
    p.add_argument("--lr",         type=float, default=LEARNING_RATE)
    p.add_argument("--lora_r",     type=int,   default=LORA_R)
    p.add_argument("--lora_alpha", type=int,   default=LORA_ALPHA)
    p.add_argument("--max_seq_len",type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--frac",    type=float, default=1.0,
                   help="Fraction of TRAINING data to use, 0<frac<=1 "
                        "(seeded; test set always stays at 100%%). Default 1.0")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_test",  type=int, default=None)
    a = p.parse_args()
    EPOCHS, BATCH_SIZE, GRAD_ACCUM = a.epochs, a.batch, a.grad_accum
    LEARNING_RATE, LORA_R, LORA_ALPHA, MAX_SEQ_LEN = a.lr, a.lora_r, a.lora_alpha, a.max_seq_len

    # Results dir named after the (train, test) file combination.
    out = _result_dir(a.train, a.test)
    print(f"\n{'═'*60}\n  LoRA FINE-TUNING — OpenBioLLM-8B\n{'═'*60}")
    print(f"  Model {MODEL_ID} | device {DEVICE} ({N_GPUS} GPU) | epochs {EPOCHS} | "
          f"batch {BATCH_SIZE} x accum {GRAD_ACCUM} | lr {LEARNING_RATE} | "
          f"LoRA r{LORA_R}/a{LORA_ALPHA}")
    print(f"  Results dir: {out}")

    print("\n[1/5] Loading data ...")
    train_s = load_jsonl(a.train)
    test_s  = load_jsonl(a.test)
    # --frac: keep a seeded fraction of the TRAINING data only (test stays full)
    if not 0.0 < a.frac <= 1.0:
        raise ValueError(f"--frac must be in (0, 1], got {a.frac}")
    if a.frac < 1.0:
        random.shuffle(train_s)
        keep = max(1, int(len(train_s) * a.frac))
        print(f"  --frac {a.frac:.3f}: using {keep:,} of {len(train_s):,} train rows")
        train_s = train_s[:keep]
    if a.n_train: random.shuffle(train_s); train_s = train_s[:a.n_train]
    if a.n_test:  random.shuffle(test_s);  test_s  = test_s[:a.n_test]
    print(f"  Train {len(train_s):,} | Test {len(test_s):,}")
    if train_s:
        ks = sorted(train_s[0].keys())
        print(f"  Detected fields in first train row: {ks}")
        print(f"  Using  instruction<-{_first_present(train_s[0], INSTRUCTION_KEYS)}  "
              f"input<-{_first_present(train_s[0], INPUT_KEYS)}  "
              f"output<-{_first_present(train_s[0], OUTPUT_KEYS)}")

    print("\n[2/5] Loading model ...")
    tok, model = load_base_model(MODEL_ID)
    print("\n[3/5] Applying LoRA ...")
    model = apply_lora(model)

    print("\n[4/5] Training ...")
    ds = MentalHealthDataset(train_s, tok, MAX_SEQ_LEN)
    nw = min(8, os.cpu_count() or 1)         # opt: more dataloader workers (plenty of RAM/cores)
    collate = functools.partial(collate_fn, pad_id=tok.pad_token_id)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate,
                        num_workers=nw, prefetch_factor=4 if nw else None,
                        persistent_workers=bool(nw), pin_memory=True)
    model = train(model, loader, EPOCHS, LEARNING_RATE, GRAD_ACCUM)

    # NOTE: trained adapter is intentionally NOT saved.

    print("\n[5/5] Evaluating ...")
    anls, results = evaluate(model, tok, test_s, EVAL_BATCH)
    print(f"\n{'═'*60}\n  OpenBioLLM-8B — ANLS {anls:.4f} ({anls*100:.2f}%)\n{'═'*60}")

    results_path = os.path.join(out, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"run_info": {"model": MODEL_ID,
                                "train_file": os.path.basename(a.train),
                                "test_file": os.path.basename(a.test),
                                "train_samples": len(ds), "test_samples": len(test_s),
                                "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                                "grad_accum": GRAD_ACCUM, "lr": LEARNING_RATE,
                                "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                                "lora_targets": LORA_TARGETS, "max_seq_len": MAX_SEQ_LEN,
                                "saved_at": datetime.now().isoformat()},
                   "metrics": {"anls": round(anls, 6)},
                   "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"  Results saved → {results_path}")

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
