"""
config.py
=========
Central configuration for the split-learning experiments.

Experimental design: 2 factors x 3 levels = 9 setups, each with 10 clients.
  Factor A — Partner Data Volume Asymmetry: symmetric | moderate | high
  Factor B — Schema Heterogeneity:          homogeneous | moderate | high

Schema heterogeneity selects which dataset feeds the clients (per user spec):
  homogeneous -> tech          (train_mh_tech_dataset / test_mh_tech_dataset)
  moderate    -> mod_hetero    (train_mod_hetero      / test_mod_hetero)
  high        -> hetero        (train_hetero          / test_hetero)

The server ends in the LLM's language-model head and produces a text answer;
quality is scored with the same ANLS pipeline used by the earlier fine-tuning
scripts (normalization unchanged).
"""

import os

# ── Backbone ────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# ── Paths ───────────────────────────────────────────────────────────────────
CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(CODE_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed", "text")  # train_/test_ JSONL splits
RESULT_DIR = os.path.join(CODE_DIR, "result_split")

# Schema level -> (train file, test file)
SCHEMA_FILES = {
    "homogeneous": ("train_mh_tech_dataset.jsonl", "test_mh_tech_dataset.jsonl"),
    "moderate":    ("train_mod_hetero.jsonl",      "test_mod_hetero.jsonl"),
    "high":        ("train_hetero.jsonl",          "test_hetero.jsonl"),
}

# ── Experimental factors ────────────────────────────────────────────────────
N_CLIENTS    = 10
VOLUME_LEVELS = ["symmetric", "moderate", "high"]
SCHEMA_LEVELS = ["homogeneous", "moderate", "high"]

# Moderately-asymmetric volume distribution (percent per client, sums to 100)
MODERATE_VOLUME_PCT = [25, 15, 12, 10, 10, 8, 7, 6, 4, 3]
# Highly-asymmetric: one dominant client + floor for the rest
HIGH_DOMINANT_PCT   = 60          # client 0 holds 60%
HIGH_MIN_SAMPLES    = 30          # each remaining client gets >= this many rows

# The 9 setups, in run order
SETUPS = [
    {"name": f"{vol}__{sch}", "volume": vol, "schema": sch}
    for vol in VOLUME_LEVELS
    for sch in SCHEMA_LEVELS
]

# ── Split-learning architecture ─────────────────────────────────────────────
# Cut layer is right AFTER the client projection head and BEFORE the server LLM
# trunk, i.e. the entire transformer body lives on the server (lightest client).
#
# PROJ_HIDDEN: bottleneck width of the projection head's residual branch.
#   The old value (16) crushed the 3072-dim hidden state through 16 dims and,
#   combined with a LayerNorm at the cut, fed the frozen trunk an
#   out-of-distribution input -> loss stuck ~4.4 -> empty generations.
#   The head is now a ZERO-INIT residual (no norm), so the bottleneck only
#   limits the learned *correction*, not the signal itself (the raw embedding
#   always passes through the residual). 256 gives the head real capacity while
#   keeping the client lightweight. Lower it later if you need a smaller client.
PROJ_HIDDEN   = 256
MAX_SEQ_LEN   = 1024     # fits long inputs (DASS up to ~960 tok) + answer
MAX_NEW_TOKENS = 64      # generation budget at eval (answers are short)
EVAL_BATCH    = 64       # prompts decoded together at eval time (adjustable)

# DIAGNOSTIC: when True the client cut is the RAW frozen embedding (no LoRA
# delta, no projection head). Train only the 24M trunk LoRA. If loss drops and
# generation becomes non-empty, the bug was in the client head/bottleneck/norm.
# If loss still sits at ~4.4 with an identity cut, the bug is in the gradient
# transport / inputs_embeds path. Set back to False for real runs.
DIAGNOSTIC_IDENTITY = False

# Print per-epoch gradient/activation stats from the training loop so you can
# see whether signal is actually flowing across the cut.
DIAGNOSTIC_GRAD_STATS = True

# LoRA (same targets/ranks as the fine-tuning scripts)
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
# Client-side LoRA target (embedding only); server-side targets (the trunk)
LORA_TARGETS_SERVER = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# ── Training ────────────────────────────────────────────────────────────────
# EPOCHS was 1 (70 steps) — far too few to tell whether the pipeline learns at
# all. Bumped so the loss has a chance to move; drop back once it's working.
EPOCHS        = 3        # rounds over each client's shard per global epoch
BATCH_SIZE    = 8        # per-client micro-batch (server processes one at a time)
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO  = 0.05
FRAC_DEFAULT  = 0.1      # default fraction of TRAIN data to use
MAX_SAMPLE    = 10000     # absolute cap on total TRAIN rows (overrides frac when set)

# ── Devices (two 96 GB GPUs) ────────────────────────────────────────────────
# Heavy server trunk -> cuda:0 ; lightweight client front-end -> cuda:1.
# Falls back gracefully to a single GPU / CPU if fewer are present.
def get_devices():
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    server = "cuda:0" if n >= 1 else "cpu"
    client = "cuda:1" if n >= 2 else server
    return server, client, n

RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a mental health assessment assistant. Answer the question "
    "concisely and directly based on the given information."
)

# Field-name aliases (datasets are uniform here, but keep the robustness)
INSTRUCTION_KEYS = ("instruction", "question", "prompt", "query", "instructions")
INPUT_KEYS       = ("input", "context", "text", "passage", "content")
OUTPUT_KEYS      = ("output", "response", "answer", "completion", "label", "target")


def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)
    return RESULT_DIR
