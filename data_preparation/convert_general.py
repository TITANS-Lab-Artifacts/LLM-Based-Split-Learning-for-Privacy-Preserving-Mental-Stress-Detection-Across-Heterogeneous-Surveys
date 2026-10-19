"""
convert_mental_health_to_jsonl.py
-----------------------------------
Converts the Mental Health Dataset (292,364 rows) into a single JSONL
file for LLaMA fine-tuning, in the same format as the OSMI, DASS, and
Student Depression scripts.

FORMAT:
  instruction — a short, generic question about one outcome
  input       — the respondent's feature values (no identity info)
  output      — the answer derived from the outcome column

DATASET OVERVIEW:
  General population mental health survey covering occupation, stress,
  habits, mood, coping ability, and social behaviour.
  All columns are categorical.

OUTCOMES (6 per row):
  1. treatment        — Has the person sought mental health treatment?
  2. Growing_Stress   — Is the person's stress growing?
  3. Coping_Struggles — Does the person struggle to cope?
  4. Work_Interest    — Does the person have interest in work?
  5. Social_Weakness  — Does the person experience social weakness?
  6. care_options     — Is the person aware of mental health care options?

FEATURES USED AS INPUT:
  Occupation, self_employed, family_history, Days_Indoors,
  Changes_Habits, Mental_Health_History, Mood_Swings,
  mental_health_interview
  — Timestamp, Gender, Country excluded (identity/demographic)

HOW TO USE:
  1. Set INPUT_FILE  → path to Mental_Health_Dataset.csv
  2. Set OUTPUT_DIR  → where to save mh_general_dataset.jsonl
  3. Run: python convert_mental_health_to_jsonl.py
"""

import json
import os
import random

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT_DIR, "data", "raw", "Mental-Health-Dataset", "Mental Health Dataset.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed", "jsonl")
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# ── Feature labels ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "Occupation":             "Occupation",
    "self_employed":          "Self-employed",
    "family_history":         "Family history of mental illness",
    "Days_Indoors":           "Days spent indoors (recent period)",
    "Changes_Habits":         "Noticed changes in habits",
    "Mental_Health_History":  "Has a mental health history",
    "Mood_Swings":            "Mood swing frequency",
    "mental_health_interview":"Would discuss mental health in a job interview",
}

# All outcome columns
OUTCOME_COLS = {
    "treatment", "Growing_Stress", "Coping_Struggles",
    "Work_Interest", "Social_Weakness", "care_options",
}

# Identity columns — excluded entirely
IDENTITY_COLS = {"Timestamp", "Gender", "Country"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def build_input(row, exclude_outcome):
    """Build input string from features, excluding the predicted outcome."""
    excluded = OUTCOME_COLS | IDENTITY_COLS | {exclude_outcome}
    parts = []
    for col, label in FEATURE_LABELS.items():
        if col in excluded:
            continue
        v = clean(row.get(col))
        if v is not None:
            parts.append(f"{label}: {v}")
    return ". ".join(parts) + "." if parts else None


# ── Sample generators ─────────────────────────────────────────────────────────

def sample_treatment(row):
    v = clean(row.get("treatment"))
    if v is None:
        return None
    input_text = build_input(row, "treatment")
    if not input_text:
        return None

    sought = v.lower() == "yes"
    fam    = clean(row.get("family_history"))
    hist   = clean(row.get("Mental_Health_History"))
    mood   = clean(row.get("Mood_Swings"))

    instructions = [
        "Has this person sought mental health treatment?",
        "Did this person pursue treatment for their mental health?",
        "Is this person currently receiving mental health treatment?",
    ]

    if sought:
        factors = []
        if fam == "Yes":
            factors.append("a family history of mental illness")
        if hist == "Yes":
            factors.append("a prior mental health history")
        if mood == "High":
            factors.append("high mood swings")
        note = f" Possible contributing factors: {', '.join(factors)}." if factors else ""
        output = f"Yes, this person has sought mental health treatment.{note}"
    else:
        output = "No, this person has not sought mental health treatment."

    return random.choice(instructions), input_text, output


def sample_growing_stress(row):
    v = clean(row.get("Growing_Stress"))
    if v is None:
        return None
    input_text = build_input(row, "Growing_Stress")
    if not input_text:
        return None

    instructions = [
        "Is this person experiencing growing stress?",
        "Has this person's stress been increasing?",
        "Does this person show signs of escalating stress?",
    ]

    days  = clean(row.get("Days_Indoors"))
    cope  = clean(row.get("Coping_Struggles"))
    mood  = clean(row.get("Mood_Swings"))

    if v == "Yes":
        context = []
        if days in ("31-60 days", "More than 2 months"):
            context.append(f"spending {days.lower()} indoors")
        if cope == "Yes":
            context.append("struggling to cope")
        if mood == "High":
            context.append("high mood swings")
        note = f" This aligns with {', '.join(context)}." if context else ""
        output = f"Yes, this person is experiencing growing stress.{note}"
    elif v == "No":
        output = "No, this person is not experiencing growing stress."
    else:
        output = "This person is uncertain whether their stress is growing."

    return random.choice(instructions), input_text, output


def sample_coping_struggles(row):
    v = clean(row.get("Coping_Struggles"))
    if v is None:
        return None
    input_text = build_input(row, "Coping_Struggles")
    if not input_text:
        return None

    instructions = [
        "Does this person struggle to cope with their mental health?",
        "Is this person having difficulty coping day-to-day?",
        "How well is this person coping with mental health challenges?",
    ]

    mood    = clean(row.get("Mood_Swings"))
    stress  = clean(row.get("Growing_Stress"))
    social  = clean(row.get("Social_Weakness"))

    if v == "Yes":
        factors = []
        if mood == "High":
            factors.append("high mood swings")
        if stress == "Yes":
            factors.append("growing stress")
        if social == "Yes":
            factors.append("social weakness")
        note = f" Associated factors: {', '.join(factors)}." if factors else ""
        output = f"Yes, this person is struggling to cope.{note}"
    else:
        output = "No, this person does not report significant coping struggles."

    return random.choice(instructions), input_text, output


def sample_work_interest(row):
    v = clean(row.get("Work_Interest"))
    if v is None:
        return None
    input_text = build_input(row, "Work_Interest")
    if not input_text:
        return None

    instructions = [
        "Does this person maintain interest in their work?",
        "Has this person lost motivation or interest in work?",
        "What is this person's level of work engagement?",
    ]

    occupation = clean(row.get("Occupation")) or "person"
    mood       = clean(row.get("Mood_Swings"))
    cope       = clean(row.get("Coping_Struggles"))

    if v == "Yes":
        output = f"Yes, this {occupation.lower()} maintains interest in their work."
    elif v == "No":
        factors = []
        if mood == "High":
            factors.append("high mood swings")
        if cope == "Yes":
            factors.append("coping difficulties")
        note = f" This may be related to {', '.join(factors)}." if factors else ""
        output = f"No, this {occupation.lower()} has lost interest in their work.{note}"
    else:
        output = f"This {occupation.lower()} is uncertain about their interest in work."

    return random.choice(instructions), input_text, output


def sample_social_weakness(row):
    v = clean(row.get("Social_Weakness"))
    if v is None:
        return None
    input_text = build_input(row, "Social_Weakness")
    if not input_text:
        return None

    instructions = [
        "Does this person experience social weakness or withdrawal?",
        "Is this person struggling with social interactions?",
        "Does this person show signs of social isolation or weakness?",
    ]

    days  = clean(row.get("Days_Indoors"))
    mood  = clean(row.get("Mood_Swings"))
    cope  = clean(row.get("Coping_Struggles"))

    if v == "Yes":
        factors = []
        if days in ("31-60 days", "More than 2 months"):
            factors.append(f"extended time indoors ({days.lower()})")
        if mood == "High":
            factors.append("high mood swings")
        if cope == "Yes":
            factors.append("coping difficulties")
        note = f" Related factors: {', '.join(factors)}." if factors else ""
        output = f"Yes, this person experiences social weakness or withdrawal.{note}"
    elif v == "No":
        output = "No, this person does not show signs of social weakness."
    else:
        output = "This person is uncertain whether they experience social weakness."

    return random.choice(instructions), input_text, output


def sample_care_options(row):
    v = clean(row.get("care_options"))
    if v is None:
        return None
    input_text = build_input(row, "care_options")
    if not input_text:
        return None

    instructions = [
        "Is this person aware of mental health care options available to them?",
        "Does this person know about mental health care resources?",
        "What is this person's awareness of available mental health care?",
    ]

    treatment = clean(row.get("treatment"))
    interview = clean(row.get("mental_health_interview"))

    if v == "Yes":
        note = " They have also sought treatment." if treatment == "Yes" else ""
        output = f"Yes, this person is aware of mental health care options available to them.{note}"
    elif v == "No":
        note = " They have not sought treatment either." if treatment == "No" else ""
        output = f"No, this person is not aware of mental health care options.{note}"
    else:
        output = "This person is unsure about what mental health care options are available to them."

    return random.choice(instructions), input_text, output


OUTCOME_GENERATORS = [
    sample_treatment,
    sample_growing_stress,
    sample_coping_struggles,
    sample_work_interest,
    sample_social_weakness,
    sample_care_options,
]


def row_to_samples(row):
    samples = []
    for gen_fn in OUTCOME_GENERATORS:
        result = gen_fn(row)
        if result:
            instruction, inp, output = result
            samples.append({
                "instruction": instruction,
                "input":       inp,
                "output":      output,
            })
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

all_samples = []
for _, row in df.iterrows():
    all_samples.extend(row_to_samples(row.to_dict()))

# ── Deduplicate generated samples ─────────────────────────────────────────────
# Some source rows collapse to identical samples once identity columns are
# dropped, so the same (input, output) text can be generated many times.
# Training on exact duplicates wastes compute and skews the model toward
# over-represented combinations. We dedup on (input, output) so instruction
# paraphrase variety is preserved while true content duplicates are removed.
# Format is unchanged (still instruction/input/output).
_raw = len(all_samples)
_seen = set()
_deduped = []
for s in all_samples:
    key = (s["input"], s["output"])
    if key in _seen:
        continue
    _seen.add(key)
    _deduped.append(s)
all_samples = _deduped
print(f"Deduplicated samples: {_raw:,} -> {len(all_samples):,} "
      f"({(1 - len(all_samples)/max(_raw,1))*100:.1f}% removed)")

random.shuffle(all_samples)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "mh_general_dataset.jsonl")

with open(out_path, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)
print(f"Input rows    : {len(df):,}")
print(f"Total samples : {len(all_samples):,}")
print(f"Avg per row   : {len(all_samples)/len(df):.1f}")
print(f"Output        : {out_path}")

print("\nOutcome distribution in source data:")
for col in ["treatment", "Growing_Stress", "Coping_Struggles",
            "Work_Interest", "Social_Weakness", "care_options"]:
    if col in df.columns:
        counts = df[col].value_counts().to_dict()
        print(f"  {col:<25} {counts}")

if all_samples:
    print("\nSample entry:")
    print(json.dumps(all_samples[0], indent=2, ensure_ascii=False))
else:
    print("No samples generated — check INPUT_FILE path.")
print("=" * 60)