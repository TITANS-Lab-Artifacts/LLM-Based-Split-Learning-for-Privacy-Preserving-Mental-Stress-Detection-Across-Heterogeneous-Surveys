"""
convert_student_depression_to_jsonl.py
---------------------------------------
Converts the Student Depression Dataset into a single JSONL file for
LLaMA fine-tuning, in the same format as the OSMI and DASS scripts.

FORMAT:
  instruction — a short, generic question about one outcome
  input       — the respondent's feature values (no identity info)
  output      — the answer derived from the outcome column

DATASET OVERVIEW (27,901 rows):
  Mix of students and working professionals from Indian cities.
  Key features: academic/work pressure, CGPA, sleep, diet, financial
  stress, suicidal thoughts, family MH history.
  Target: Depression (0 = No, 1 = Yes)

OUTCOMES (5 per row):
  1. Depression status (primary binary outcome)
  2. Whether the person has had suicidal thoughts
  3. Sleep adequacy assessment
  4. Overall stress burden (composite of academic + financial + work pressure)
  5. Lifestyle risk profile (diet + sleep + study hours combined)

FEATURES USED AS INPUT:
  Profession, Academic Pressure, Work Pressure, CGPA, Study Satisfaction,
  Job Satisfaction, Sleep Duration, Dietary Habits, Degree,
  Work/Study Hours, Financial Stress, Family History of Mental Illness
  — City, Gender, Age, id excluded (identity/demographic)

HOW TO USE:
  1. Set INPUT_FILE  → path to Student_Depression_Dataset.csv
  2. Set OUTPUT_DIR  → where to save the JSONL
  3. Run: python convert_student_depression_to_jsonl.py
"""

import json
import os
import random

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(ROOT_DIR, "data", "raw", "Student-Depression", "Student Depression Dataset.csv")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "data", "processed", "jsonl")
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# ── Feature labels (human-readable) ──────────────────────────────────────────
FEATURE_LABELS = {
    "Profession":                       "Profession",
    "Academic Pressure":                "Academic pressure (0=none, 5=extreme)",
    "Work Pressure":                    "Work pressure (0=none, 5=extreme)",
    "CGPA":                             "CGPA (academic performance, 0-10 scale)",
    "Study Satisfaction":               "Study satisfaction (0=none, 5=very high)",
    "Job Satisfaction":                 "Job satisfaction (0=none, 5=very high)",
    "Sleep Duration":                   "Sleep duration",
    "Dietary Habits":                   "Dietary habits",
    "Degree":                           "Degree being pursued",
    "Work/Study Hours":                 "Daily work/study hours",
    "Financial Stress":                 "Financial stress (1=low, 5=extreme)",
    "Family History of Mental Illness": "Family history of mental illness",
}

# Columns that are outcomes — excluded from input when predicting that outcome
OUTCOME_COLS = {
    "Depression",
    "Have you ever had suicidal thoughts ?",
}

# Identity columns — excluded entirely from input and output
IDENTITY_COLS = {"id", "Gender", "Age", "City"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(v):
    """Return string value or None if missing/invalid."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def build_input_text(row, exclude_outcome):
    """
    Build the input string from feature columns.
    Excludes the specific outcome column being predicted, all other
    outcome columns, and all identity columns.
    """
    parts = []
    excluded = OUTCOME_COLS | IDENTITY_COLS | {exclude_outcome}

    for col, label in FEATURE_LABELS.items():
        if col in excluded:
            continue
        v = clean(row.get(col))
        if v is None:
            continue
        # Format numeric pressure/satisfaction scales with context
        parts.append(f"{label}: {v}")

    return ". ".join(parts) + "." if parts else None


# ── Sample generators — one per outcome ──────────────────────────────────────

def sample_depression(row):
    v = clean(row.get("Depression"))
    if v is None:
        return None

    input_text = build_input_text(row, exclude_outcome="Depression")
    if not input_text:
        return None

    depressed = str(v) == "1"
    profession = clean(row.get("Profession")) or "individual"
    fam_history = clean(row.get("Family History of Mental Illness"))
    sleep = clean(row.get("Sleep Duration"))
    acad = safe_float(row.get("Academic Pressure"))
    fin = safe_float(row.get("Financial Stress"))

    instructions = [
        "Does this person show signs of depression?",
        "Is this student or professional experiencing depression?",
        "Based on this profile, does this person have depression?",
        "What is the depression status of this person?",
    ]

    if depressed:
        context_parts = []
        if fam_history == "Yes":
            context_parts.append("a family history of mental illness")
        if sleep in ("Less than 5 hours", "5-6 hours"):
            context_parts.append("insufficient sleep")
        if acad and acad >= 4:
            context_parts.append("high academic pressure")
        if fin and fin >= 4:
            context_parts.append("significant financial stress")

        context = f" Contributing factors include {', '.join(context_parts)}." if context_parts else ""
        output = f"Yes, this {profession.lower()} is experiencing depression.{context}"
    else:
        output = f"No, this {profession.lower()} does not show signs of depression."

    return random.choice(instructions), input_text, output


def sample_suicidal_thoughts(row):
    v = clean(row.get("Have you ever had suicidal thoughts ?"))
    if v is None:
        return None

    input_text = build_input_text(row, exclude_outcome="Have you ever had suicidal thoughts ?")
    if not input_text:
        return None

    has_thoughts = v.lower() == "yes"
    depression = clean(row.get("Depression"))
    depressed = str(depression) == "1" if depression else None

    instructions = [
        "Has this person ever experienced suicidal thoughts?",
        "Does this person's profile suggest a history of suicidal ideation?",
        "Has this student or professional ever had suicidal thoughts?",
    ]

    if has_thoughts:
        context = " They are also currently experiencing depression." if depressed else ""
        output = f"Yes, this person has experienced suicidal thoughts.{context}"
    else:
        output = "No, this person has not reported experiencing suicidal thoughts."

    return random.choice(instructions), input_text, output


def sample_sleep_adequacy(row):
    sleep = clean(row.get("Sleep Duration"))
    if not sleep:
        return None

    # Exclude sleep duration from input since it IS the outcome
    parts = []
    excluded = OUTCOME_COLS | IDENTITY_COLS | {"Sleep Duration"}
    for col, label in FEATURE_LABELS.items():
        if col in excluded:
            continue
        v = clean(row.get(col))
        if v is not None:
            parts.append(f"{label}: {v}")
    input_text = ". ".join(parts) + "." if parts else None
    if not input_text:
        return None

    instructions = [
        "How adequate is this person's sleep?",
        "What is this person's sleep pattern and is it sufficient?",
        "Assess the sleep quality for this student or professional.",
    ]

    adequacy_map = {
        "Less than 5 hours": ("inadequate", "severely sleep-deprived, getting less than 5 hours per night"),
        "5-6 hours":         ("insufficient", "getting only 5-6 hours of sleep, below the recommended 7-9 hours"),
        "7-8 hours":         ("adequate", "getting a healthy 7-8 hours of sleep per night"),
        "More than 8 hours": ("excessive", "sleeping more than 8 hours, which may indicate fatigue or low motivation"),
        "Others":            ("unspecified", "reporting an unspecified sleep pattern"),
    }

    level, description = adequacy_map.get(sleep, ("unknown", sleep))
    work_hours = safe_float(row.get("Work/Study Hours"))
    hours_note = f" They study/work {work_hours:.0f} hours daily." if work_hours else ""

    output = f"This person's sleep is {level} — they are {description}.{hours_note}"
    return random.choice(instructions), input_text, output


def sample_stress_burden(row):
    acad = safe_float(row.get("Academic Pressure"))
    fin  = safe_float(row.get("Financial Stress"))
    work = safe_float(row.get("Work Pressure"))

    # Need at least two stress dimensions
    available = [x for x in [acad, fin, work] if x is not None]
    if len(available) < 2:
        return None

    # Exclude stress columns from input
    parts = []
    excluded = OUTCOME_COLS | IDENTITY_COLS | {"Academic Pressure", "Work Pressure", "Financial Stress"}
    for col, label in FEATURE_LABELS.items():
        if col in excluded:
            continue
        v = clean(row.get(col))
        if v is not None:
            parts.append(f"{label}: {v}")
    input_text = ". ".join(parts) + "." if parts else None
    if not input_text:
        return None

    instructions = [
        "What is the overall stress burden on this person?",
        "How much stress is this student or professional under?",
        "Assess the combined stress level for this person.",
    ]

    composite = sum(available) / len(available)
    if composite <= 1.5:
        level = "low"
    elif composite <= 3.0:
        level = "moderate"
    elif composite <= 4.0:
        level = "high"
    else:
        level = "very high"

    stress_parts = []
    if acad is not None:
        stress_parts.append(f"academic pressure: {acad:.0f}/5")
    if work is not None:
        stress_parts.append(f"work pressure: {work:.0f}/5")
    if fin is not None:
        stress_parts.append(f"financial stress: {fin:.0f}/5")

    output = (
        f"This person faces a {level} overall stress burden "
        f"({'; '.join(stress_parts)}; average: {composite:.1f}/5)."
    )
    return random.choice(instructions), input_text, output


def sample_lifestyle_risk(row):
    diet  = clean(row.get("Dietary Habits"))
    sleep = clean(row.get("Sleep Duration"))
    hours = safe_float(row.get("Work/Study Hours"))

    available = [x for x in [diet, sleep, hours] if x is not None]
    if len(available) < 2:
        return None

    # Exclude lifestyle columns from input
    parts = []
    excluded = OUTCOME_COLS | IDENTITY_COLS | {"Dietary Habits", "Sleep Duration", "Work/Study Hours"}
    for col, label in FEATURE_LABELS.items():
        if col in excluded:
            continue
        v = clean(row.get(col))
        if v is not None:
            parts.append(f"{label}: {v}")
    input_text = ". ".join(parts) + "." if parts else None
    if not input_text:
        return None

    instructions = [
        "What is the lifestyle risk profile of this person?",
        "How healthy is this person's lifestyle based on their habits?",
        "Assess the lifestyle risk factors for this student or professional.",
    ]

    risk_factors = []
    protective = []

    if diet == "Unhealthy":
        risk_factors.append("unhealthy diet")
    elif diet == "Moderate":
        risk_factors.append("moderately healthy diet")
    elif diet in ("Healthy",):
        protective.append("healthy diet")

    if sleep == "Less than 5 hours":
        risk_factors.append("severely insufficient sleep (<5 hours)")
    elif sleep == "5-6 hours":
        risk_factors.append("insufficient sleep (5-6 hours)")
    elif sleep == "7-8 hours":
        protective.append("adequate sleep (7-8 hours)")
    elif sleep == "More than 8 hours":
        risk_factors.append("excessive sleep (>8 hours)")

    if hours is not None:
        if hours >= 10:
            risk_factors.append(f"excessive daily study/work hours ({hours:.0f}h)")
        elif hours <= 4:
            risk_factors.append(f"very low daily study/work hours ({hours:.0f}h)")
        else:
            protective.append(f"reasonable daily study/work hours ({hours:.0f}h)")

    if not risk_factors and not protective:
        return None

    if len(risk_factors) >= 2:
        level = "high risk"
    elif len(risk_factors) == 1:
        level = "moderate risk"
    else:
        level = "low risk"

    parts_out = []
    if risk_factors:
        parts_out.append(f"risk factors: {', '.join(risk_factors)}")
    if protective:
        parts_out.append(f"protective factors: {', '.join(protective)}")

    output = f"This person has a {level} lifestyle profile ({'; '.join(parts_out)})."
    return random.choice(instructions), input_text, output


OUTCOME_GENERATORS = [
    sample_depression,
    sample_suicidal_thoughts,
    sample_sleep_adequacy,
    sample_stress_burden,
    sample_lifestyle_risk,
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
out_path = os.path.join(OUTPUT_DIR, "mh_student_dataset.jsonl")

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

print("\nOutcome sample counts:")
counts = {}
for s in all_samples:
    key = s["instruction"]
    counts[key] = counts.get(key, 0) + 1
# Collapse by first few words
collapsed = {}
for k, v in counts.items():
    short = k[:52]
    collapsed[short] = collapsed.get(short, 0) + v
for k, v in sorted(collapsed.items(), key=lambda x: -x[1]):
    print(f"  {k:<54} {v:>7,}")

print("\nDepression distribution in source data:")
print(df["Depression"].value_counts().to_string())

if all_samples:
    print("\nSample entry:")
    print(json.dumps(all_samples[0], indent=2, ensure_ascii=False))
else:
    print("No samples generated — check INPUT_FILE path.")
print("=" * 60)
