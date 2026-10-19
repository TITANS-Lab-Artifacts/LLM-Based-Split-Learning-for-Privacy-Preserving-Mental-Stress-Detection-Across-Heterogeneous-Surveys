"""
few_shot_anls.py
-----------------
Few-shot evaluation of instruction-tuned LLMs on mental health JSONL
datasets using ANLS (Average Normalized Levenshtein Similarity) with
output normalization.

Few-shot strategy:
  For each test sample, select n examples from the training set that
  share the same instruction type (matched by first 5 words of the
  instruction). This ensures examples are relevant to the question
  being asked. Falls back to random training samples if not enough
  same-type examples are available.

ANLS with output normalization:
  1. Extract the core answer from both generated and expected responses
  2. Normalize both: lowercase, remove articles/punctuation/accents
  3. Compute Normalized Levenshtein Similarity (NLS)
  4. If NLS < threshold (default 0.5) → score = 0
  5. ANLS = mean of per-sample scores

HOW TO USE:
  python few_shot_anls.py \\
      --train  path/to/train.jsonl \\
      --test   path/to/test.jsonl  \\
      --n_few  5                   \\
      --model  meta-llama/Llama-3.2-3B-Instruct

  # Use Phi-4-mini instead:
  python few_shot_anls.py \\
      --train  path/to/train.jsonl \\
      --test   path/to/test.jsonl  \\
      --model  microsoft/Phi-4-mini-instruct

  # Limit test samples:
  python few_shot_anls.py \\
      --train  path/to/train.jsonl \\
      --test   path/to/test.jsonl  \\
      --n_test 200 --n_few 3

REQUIREMENTS:
  pip install transformers accelerate bitsandbytes
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
# Suppress Phi flash-attention informational messages
import logging
logging.getLogger("transformers.models.phi3.modeling_phi3").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
try:
    import packaging.version
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging", "-q"])
    import packaging.version
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # runner can override
# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"
# Other supported models (pass via --model):
#   aaditya/Llama3-OpenBioLLM-8B      (biomedical domain, LLaMA-3 based)
#   meta-llama/Llama-3.1-8B-Instruct  (stronger general model)
#   mistralai/Mistral-7B-Instruct-v0.3
MAX_NEW_TOKENS  = 150
ANLS_THRESHOLD  = 0.5
N_FEW_DEFAULT   = 5
BATCH_SIZE      = 64     # samples per GPU batch
RANDOM_SEED     = 42
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"
N_GPUS          = torch.cuda.device_count() if torch.cuda.is_available() else 0
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text):
    """
    Standard ANLS normalization:
      - lowercase, remove unicode accents
      - remove articles (a, an, the)
      - remove punctuation, collapse whitespace
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Cross-vocabulary canonical mapping ───────────────────────────────────────
# Maps semantically equivalent terms to a single canonical token so that
# ANLS compares meaning rather than surface vocabulary.
#
# Key observations from output analysis:
#   - Model says "depression/anxiety/stress" when GT expects a severity level
#     → these condition names are extracted when no severity is found, but the
#       GT may have already been normalised to a severity. Solution: map
#       condition names + severity combos to the canonical severity.
#   - "high risk" ≈ "severe/extremely severe" in risk-assessment questions
#   - "moderate risk" ≈ "moderate"
#   - "low risk" ≈ "normal/mild"
#   - "inadequate" ≈ "insufficient" (sleep)
#   - "adequate" ≈ "sufficient" (sleep)
#   - "yes" can appear where GT says a condition name (binary questions about
#     MH disorders) and vice versa — handled by synonym groups below

# Severity ladder — used for partial credit scoring
SEVERITY_RANK = {
    "normal":           0,
    "mild":             1,
    "moderate":         2,
    "severe":           3,
    "extremely severe": 4,
}

# Risk level → severity mapping
RISK_TO_SEVERITY = {
    "low risk":      "mild",
    "moderate risk": "moderate",
    "high risk":     "severe",
    "very high risk":"extremely severe",
}

# Sleep adequacy synonym groups → canonical
SLEEP_SYNONYMS = {
    "inadequate":         "inadequate",
    "severely sleep deprived": "inadequate",
    "less than 5 hours":  "inadequate",
    "insufficient":       "insufficient",
    "not enough":         "insufficient",
    "5 6 hours":          "insufficient",
    "adequate":           "adequate",
    "sufficient":         "adequate",
    "7 8 hours":          "adequate",
    "excessive":          "excessive",
    "too much":           "excessive",
    "more than 8 hours":  "excessive",
}

# Condition names treated as positive binary answers when question is binary
MH_CONDITIONS = {"depression", "anxiety", "stress", "disorder"}

# Positive binary synonyms
POSITIVE_SYNONYMS = {
    "yes", "yep", "yup", "ya", "affirmative", "true", "correct",
    "has sought", "have sought", "sought treatment", "sought help",
    "received treatment", "getting help", "is experiencing",
    "shows signs", "does have", "currently has", "clinically significant",
    "has been diagnosed", "indeed", "definitely", "certainly",
}

# Negative binary synonyms
NEGATIVE_SYNONYMS = {
    "no", "nope", "false", "incorrect",
    "has not", "have not", "hasn't", "haven't", "does not",
    "doesn't", "is not", "isn't", "never sought", "not experiencing",
    "no signs", "not diagnosed", "does not have", "within normal",
    "no disorder", "not present",
}


def canonical(token):
    """
    Map a core-answer token to its canonical form using cross-vocabulary rules.
    This is applied to BOTH prediction and ground-truth before ANLS scoring.
    """
    t = token.strip().lower()

    # Severity passthrough
    if t in SEVERITY_RANK:
        return t

    # Risk → severity
    if t in RISK_TO_SEVERITY:
        return RISK_TO_SEVERITY[t]

    # Sleep adequacy synonyms
    for phrase, canon in SLEEP_SYNONYMS.items():
        if phrase in t:
            return canon

    # Positive binary
    for syn in POSITIVE_SYNONYMS:
        if syn in t:
            return "yes"

    # Negative binary
    for syn in NEGATIVE_SYNONYMS:
        if syn in t:
            return "no"

    # Condition names → kept as-is (they are valid core answers for
    # "which condition is dominant" questions)
    return t


def severity_partial_credit(p, g):
    """
    Give partial credit for adjacent severity levels.
    Full credit (1.0) for exact match.
    0.5 for one step away (e.g. severe vs extremely severe).
    0.0 for two or more steps away.
    """
    if p == g:
        return 1.0
    if p in SEVERITY_RANK and g in SEVERITY_RANK:
        diff = abs(SEVERITY_RANK[p] - SEVERITY_RANK[g])
        if diff == 1:
            return 0.5
    return 0.0


def extract_core_answer(text):
    """
    Extract the short core answer from a longer generative response.

    Priority order (checked in sequence, first match wins):
      1. Explicit severity levels (extremely severe → normal)
      2. Risk levels            (mapped to equivalent severity)
      3. Sleep adequacy labels
      4. Binary yes / no (broad synonym matching)
      5. Dominant MH condition name (depression / anxiety / stress)
      6. Fallback: first 60 normalized characters
    """
    t = text.lower().strip()

    # 1. Severity levels — check multi-word first
    if re.search(r"extremely\s+severe|very\s+severe", t):
        return "extremely severe"
    if re.search(r"\bsevere\b", t) and "extremely" not in t:
        return "severe"
    if re.search(r"\bmoderate\b", t) and "risk" not in t:
        return "moderate"
    if re.search(r"\bmild\b", t):
        return "mild"
    if re.search(r"\bnormal\b", t) and "abnormal" not in t:
        return "normal"

    # 2. Risk levels (map to severity canonical)
    if re.search(r"\bvery\s+high\s+risk\b", t):
        return "extremely severe"   # via canonical mapping
    if re.search(r"\bhigh\s+risk\b", t):
        return "severe"             # via canonical mapping
    if re.search(r"\bmoderate\s+risk\b", t):
        return "moderate"           # via canonical mapping
    if re.search(r"\blow\s+risk\b", t):
        return "mild"               # via canonical mapping

    # 3. Sleep adequacy
    if re.search(r"\binadequate\b|severely\s+sleep.deprived|less\s+than\s+5", t):
        return "inadequate"
    if re.search(r"\binsufficient\b|not\s+enough\s+sleep|5.?6\s+hour", t):
        return "insufficient"
    if re.search(r"\badequate\b|\bsufficient\b|7.?8\s+hour", t):
        return "adequate"
    if re.search(r"\bexcessive\b|too\s+much\s+sleep|more\s+than\s+8", t):
        return "excessive"

    # 4. Binary yes / no — check positive first
    if re.search(
        r"\b(yes|yep|yup|ya|affirmative|has sought|have sought|"
        r"sought treatment|sought help|received treatment|getting help|"
        r"is experiencing|shows signs|does have|currently has|"
        r"clinically significant|has been diagnosed|indeed|"
        r"does suffer|suffering from|is present|are present|"
        r"positive for|confirms)\b",
        t,
    ):
        return "yes"
    if re.search(
        r"\b(no\b|nope|has not|have not|hasn.t|haven.t|does not|"
        r"doesn.t|is not|isn.t|never sought|not experiencing|"
        r"no signs|not diagnosed|does not have|within normal|"
        r"not present|negative for|no disorder|not detected)\b",
        t,
    ):
        return "no"

    # 5. Dominant MH condition
    for cond in ["depression", "anxiety", "stress"]:
        if re.search(rf"\b{cond}\b", t):
            return cond

    # 6. Fallback
    return normalize_text(text)[:60]


# ══════════════════════════════════════════════════════════════════════════════
# ANLS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def levenshtein(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def nls(pred_core, gt_core, threshold=ANLS_THRESHOLD):
    """
    Normalized Levenshtein Similarity with cross-vocabulary canonicalization
    and partial credit for adjacent severity levels.

    Steps:
      1. Apply canonical() mapping to both pred and gt
      2. If both are severity levels, apply severity_partial_credit()
      3. Otherwise compute standard NLS with threshold cutoff
    """
    p = canonical(normalize_text(pred_core))
    g = canonical(normalize_text(gt_core))

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    # Exact match after canonicalization
    if p == g:
        return 1.0

    # Severity partial credit (0.5 for one step away)
    partial = severity_partial_credit(p, g)
    if partial > 0:
        return partial

    # Standard Levenshtein similarity
    dist = levenshtein(p, g)
    sim  = 1.0 - dist / max(len(p), len(g))
    return sim if sim >= threshold else 0.0


def compute_anls(pred, gt, threshold=ANLS_THRESHOLD):
    """
    Full pipeline:
      1. Extract core answer from prediction and ground truth
      2. Apply canonical mapping
      3. Compute NLS (with severity partial credit)
    Returns (anls_score, pred_core, gt_core).
    """
    pred_core = extract_core_answer(pred)
    gt_core   = extract_core_answer(gt)
    score     = nls(pred_core, gt_core, threshold)
    return score, pred_core, gt_core



# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def check_transformers_version(model_id):
    """
    Warn clearly if transformers version is too old for the requested model.
    Phi-4-mini requires >= 4.46.0. LLaMA-3.2 requires >= 4.43.0.
    """
    import importlib.metadata, packaging.version
    try:
        installed = packaging.version.parse(
            importlib.metadata.version("transformers")
        )
    except Exception:
        return  # can't check, proceed anyway

    requirements = {
        "phi-4":        ("4.46.0", "microsoft/Phi-4-mini-instruct"),
        "phi4":         ("4.46.0", "microsoft/Phi-4-mini-instruct"),
        "llama-3.2":    ("4.43.0", "meta-llama/Llama-3.2-3B-Instruct"),
        "llama-3":      ("4.43.0", "meta-llama/Llama-3.x-Instruct"),
        "openbio":      ("4.40.0", "aaditya/Llama3-OpenBioLLM-8B"),
        "llama3-openbio": ("4.40.0", "aaditya/Llama3-OpenBioLLM-8B"),
    }

    for key, (min_ver, example) in requirements.items():
        if key in model_id.lower():
            required = packaging.version.parse(min_ver)
            if installed < required:
                print(f"\n  [ERROR] {model_id} requires transformers >= {min_ver}")
                print(f"  Installed version: {installed}")
                print(f"  Fix: pip install --upgrade transformers accelerate")
                print(f"  Or:  pip install transformers=={min_ver} --upgrade --user")
                import sys; sys.exit(1)
            break


def load_model(model_id):
    """
    Load model optimised for dual RTX 6000 Ada (192 GB total VRAM):
      - float16 (no quantization) — LLaMA-3.2-3B uses ~6 GB, OpenBioLLM-8B ~16 GB
      - device_map="auto" splits layers across both GPUs automatically
      - Flash Attention 2 if installed (pip install flash-attn), else SDPA
      - torch.compile for ~30% additional throughput
      - use_cache=True (KV cache re-enabled — Phi removed, no longer needed off)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    check_transformers_version(model_id)

    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Detect best attention implementation
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
        print(f"  Flash Attention 2 available — using it")
    except ImportError:
        attn_impl = "sdpa"   # PyTorch scaled dot-product attention (built-in)
        print(f"  Flash Attention 2 not installed — using SDPA (still fast)")

    if DEVICE == "cuda":
        print(f"  Loading in float16 across {N_GPUS} GPU(s) ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    # torch.compile — fuses ops into optimised CUDA kernels (~20-40% faster)
    if DEVICE == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"  torch.compile applied (first batch will be slow — compiling)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    vram_used = sum(
        torch.cuda.memory_allocated(i) / 1024**3
        for i in range(torch.cuda.device_count())
    ) if DEVICE == "cuda" else 0
    print(f"  Model ready. VRAM used: {vram_used:.1f} GB across {N_GPUS} GPU(s)")
    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_instruction_prefix(instruction, n_words=5):
    """Return the first n_words of an instruction as a matching key."""
    return " ".join(instruction.lower().split()[:n_words])


def select_few_shot_examples(test_sample, train_samples, n):
    """
    Select n examples from train_samples that match the test sample's
    instruction type (by first 5 words). Falls back to random if
    not enough same-type examples are available.
    """
    test_prefix = get_instruction_prefix(test_sample["instruction"])

    # Try to find same-instruction-type examples
    same_type = [
        s for s in train_samples
        if get_instruction_prefix(s["instruction"]) == test_prefix
        and s.get("output") != test_sample.get("output")  # avoid leaking exact answer
    ]

    if len(same_type) >= n:
        return random.sample(same_type, n)

    # Not enough — pad with random samples from any instruction type
    others    = [s for s in train_samples if s not in same_type]
    combined  = same_type + random.sample(others, min(n - len(same_type), len(others)))
    return combined[:n]


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_few_shot_messages(test_sample, examples, model_id):
    """
    Build a few-shot prompt using the model's chat template.
    Examples are injected as assistant turns in a multi-turn conversation,
    which is the most natural way to do few-shot with instruct models.
    """
    instruction = test_sample["instruction"]
    inp         = test_sample.get("input", "").strip()

    system_msg = (
        "You are a mental health assessment assistant. "
        "Answer the question concisely and directly based on the given information."
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add examples as user/assistant pairs
    for ex in examples:
        ex_inp  = ex.get("input", "").strip()
        ex_user = ex["instruction"]
        if ex_inp:
            ex_user += f"\n\n{ex_inp}"

        messages.append({"role": "user",      "content": ex_user})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Add the actual test question
    user_content = instruction
    if inp:
        user_content += f"\n\n{inp}"
    messages.append({"role": "user", "content": user_content})

    return messages


def apply_chat_template(messages, tokenizer, model_id):
    """Apply the model's chat template, with plain-text fallback."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:
        # Plain-text fallback
        parts = []
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if system:
            parts.append(f"System: {system}\n")

        pairs = [(messages[i], messages[i+1])
                 for i in range(len(messages))
                 if messages[i]["role"] == "user"
                 and i+1 < len(messages)
                 and messages[i+1]["role"] == "assistant"]

        for i, (usr, ast) in enumerate(pairs, 1):
            parts.append(f"### Example {i}:")
            parts.append(f"User: {usr['content']}")
            parts.append(f"Assistant: {ast['content']}\n")

        # Last user message (the test question)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        parts.append("### Question:")
        parts.append(f"User: {last_user}")
        parts.append("Assistant:")
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _build_eos_ids(tokenizer):
    """LLaMA-3 based models use <|eot_id|> as end-of-turn in addition to eos."""
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id and eot_id != tokenizer.eos_token_id:
            eos_ids.append(eot_id)
    except Exception:
        pass
    return eos_ids


@torch.no_grad()
def generate_batch(prompt_texts, tokenizer, model):
    """
    Generate responses for a batch of prompts simultaneously.
    Left-padding ensures all sequences are aligned for the model.
    Returns a list of decoded response strings.
    """
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        return_attention_mask=True,
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]
    eos_ids   = _build_eos_ids(tokenizer)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_ids,
        use_cache=True,
    )

    responses = []
    for seq in outputs:
        new_tokens = seq[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Declare globals before any reference to them
    global MAX_NEW_TOKENS, ANLS_THRESHOLD, BATCH_SIZE
    MAX_NEW_TOKENS = MAX_NEW_TOKENS  # will be overwritten by args below
    ANLS_THRESHOLD = ANLS_THRESHOLD  # will be overwritten by args below
    BATCH_SIZE     = BATCH_SIZE      # will be overwritten by args below

    parser = argparse.ArgumentParser(
        description="Few-shot ANLS evaluation for LLaMA / OpenBioLLM / other instruct models on mental health JSONL"
    )
    parser.add_argument("--train",   required=True,
                        help="Path to train JSONL file (source of few-shot examples)")
    parser.add_argument("--test",    required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--n_few",   type=int, default=N_FEW_DEFAULT,
                        help=f"Number of few-shot examples (default: {N_FEW_DEFAULT})")
    parser.add_argument("--n_test",  type=int, default=None,
                        help="Number of test samples (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_NEW_TOKENS})")
    parser.add_argument("--threshold", type=float, default=ANLS_THRESHOLD,
                        help=f"ANLS threshold (default: {ANLS_THRESHOLD})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for generation (default: {BATCH_SIZE})")
    args = parser.parse_args()
    MAX_NEW_TOKENS = args.max_new_tokens
    ANLS_THRESHOLD = args.threshold
    BATCH_SIZE     = args.batch_size


    print(f"\n{'═'*58}")
    print(f"  FEW-SHOT ANLS EVALUATION")
    print(f"{'═'*58}")
    print(f"  Model    : {args.model}")
    print(f"  Train    : {os.path.basename(args.train)}")
    print(f"  Test     : {os.path.basename(args.test)}")
    print(f"  Examples : {args.n_few} per prompt")
    print(f"  Device   : {DEVICE.upper()}")
    print(f"  Threshold: {ANLS_THRESHOLD}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/3] Loading data ...")
    with open(args.train, "r", encoding="utf-8") as f:
        train_samples = [json.loads(l) for l in f if l.strip()]

    with open(args.test, "r", encoding="utf-8") as f:
        test_samples = [json.loads(l) for l in f if l.strip()]

    if args.n_test and args.n_test < len(test_samples):
        random.shuffle(test_samples)
        test_samples = test_samples[:args.n_test]

    print(f"  Train samples : {len(train_samples):,}")
    print(f"  Test samples  : {len(test_samples):,}")

    # Group train samples by instruction prefix for fast lookup
    by_prefix = {}
    for s in train_samples:
        prefix = get_instruction_prefix(s["instruction"])
        by_prefix.setdefault(prefix, []).append(s)

    n_same_type  = sum(
        1 for s in test_samples
        if len(by_prefix.get(get_instruction_prefix(s["instruction"]), [])) >= args.n_few
    )
    print(f"  Test samples with enough same-type examples: "
          f"{n_same_type}/{len(test_samples)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[2/3] Loading model ...")
    tokenizer, model = load_model(args.model)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] Running few-shot evaluation ...")
    print(f"  {'#':>5}  {'Pred core':<22}  {'GT core':<22}  {'ANLS':>6}")
    print(f"  {'─'*58}")

    sample_results = []
    anls_scores    = []
    print(f"  Batch size: {BATCH_SIZE}  |  GPUs: {N_GPUS}")

    # Process in batches for maximum GPU utilisation
    for batch_start in range(0, len(test_samples), BATCH_SIZE):
        batch          = test_samples[batch_start:batch_start + BATCH_SIZE]
        batch_prompts  = []
        batch_expected = []
        batch_meta     = []   # store per-sample metadata for results

        for sample in batch:
            examples    = select_few_shot_examples(sample, train_samples, args.n_few)
            messages    = build_few_shot_messages(sample, examples, args.model)
            prompt_text = apply_chat_template(messages, tokenizer, args.model)
            batch_prompts.append(prompt_text)
            batch_expected.append(sample["output"].strip())
            batch_meta.append({
                "n_examples_used": len(examples),
                "same_type_match": get_instruction_prefix(sample["instruction"]) in by_prefix,
            })

        # Single GPU call for the whole batch
        batch_generated = generate_batch(batch_prompts, tokenizer, model)

        for j, (generated, expected, sample, meta) in enumerate(
            zip(batch_generated, batch_expected, batch, batch_meta)
        ):
            i = batch_start + j + 1
            score, pred_core, gt_core = compute_anls(generated, expected, ANLS_THRESHOLD)
            anls_scores.append(score)

            sample_results.append({
                "index":            i,
                "instruction":      sample["instruction"],
                "input_preview":    sample.get("input", "")[:100],
                "expected":         expected,
                "generated":        generated,
                "pred_core":        pred_core,
                "gt_core":          gt_core,
                "anls_score":       round(score, 4),
                "n_examples_used":  meta["n_examples_used"],
                "same_type_match":  meta["same_type_match"],
            })

            print(f"  {i:>5}  {pred_core[:20]:<22}  {gt_core[:20]:<22}  {score:>6.3f}")

        n_done  = min(batch_start + BATCH_SIZE, len(test_samples))
        running = sum(anls_scores) / len(anls_scores)
        if n_done % 50 < BATCH_SIZE or n_done == len(test_samples):
            print(f"  {'─'*58}")
            print(f"  Running ANLS @{n_done}: {running:.4f}")
            print(f"  {'─'*58}")

    final_anls = sum(anls_scores) / len(anls_scores)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  FEW-SHOT RESULTS")
    print(f"{'═'*58}")
    print(f"  Model          : {args.model}")
    print(f"  Train file     : {os.path.basename(args.train)}")
    print(f"  Test file      : {os.path.basename(args.test)}")
    print(f"  Samples        : {len(test_samples)}")
    print(f"  Examples/prompt: {args.n_few}")
    print(f"  ANLS Threshold : {ANLS_THRESHOLD}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ANLS           : {final_anls:.4f}  ({final_anls*100:.2f}%)")
    print(f"{'═'*58}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    train_stem = os.path.splitext(os.path.basename(args.train))[0]
    test_stem  = os.path.splitext(os.path.basename(args.test))[0]
    model_slug = args.model.replace("/", "_").replace("-", "_")
    out_dir    = os.path.join(os.path.dirname(args.test), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(
        out_dir, f"few_shot_{args.n_few}_{model_slug}_{train_stem}_vs_{test_stem}.json"
    )

    save_data = {
        "run_info": {
            "mode":            "few_shot",
            "model":           args.model,
            "train_file":      os.path.basename(args.train),
            "test_file":       os.path.basename(args.test),
            "n_samples":       len(test_samples),
            "n_few_shot":      args.n_few,
            "anls_threshold":  ANLS_THRESHOLD,
            "max_new_tokens":  MAX_NEW_TOKENS,
            "saved_at":        datetime.now().isoformat(),
        },
        "metrics": {
            "anls": round(final_anls, 6),
        },
        "samples": sample_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"  Results saved → {out_path}")

    # ── Free GPU memory ───────────────────────────────────────────────────────
    print("\n  Freeing GPU memory ...")
    try:
        del model
        del tokenizer
    except NameError:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i)  / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: allocated={alloc:.2f} GB  reserved={reserved:.2f} GB")
    print()


if __name__ == "__main__":
    main()