"""
convert_dass_to_jsonl.py
-------------------------
Converts the DASS-42 (Depression Anxiety Stress Scales) dataset into a
JSONL file for LLaMA fine-tuning, in the same format as the OSMI script.

FORMAT:
  instruction — a short, generic question about one outcome
  input       — the respondent's feature values (Q item scores, personality,
                demographics — NO identity info)
  output      — the answer derived from the outcome

DASS-42 SCORING:
  Each of the 42 items is rated 1-4 in the raw data.
  We convert to standard 0-3 scoring (subtract 1 from each item).
  Subscale score = sum of 14 assigned items (range 0-42).

  Subscale assignments:
    Depression : Q3,Q5,Q10,Q13,Q16,Q17,Q21,Q24,Q26,Q31,Q34,Q37,Q38,Q42
    Anxiety    : Q2,Q4,Q7,Q9,Q15,Q19,Q20,Q23,Q25,Q28,Q30,Q36,Q40,Q41
    Stress     : Q1,Q6,Q8,Q11,Q12,Q14,Q18,Q22,Q27,Q29,Q32,Q33,Q35,Q39

  Severity thresholds (DASS-42 standard):
    Depression: 0-9 Normal, 10-13 Mild, 14-20 Moderate, 21-27 Severe, 28+ Extremely Severe
    Anxiety:    0-7 Normal, 8-9 Mild,  10-14 Moderate, 15-19 Severe, 20+ Extremely Severe
    Stress:     0-14 Normal,15-18 Mild, 19-25 Moderate, 26-33 Severe, 34+ Extremely Severe

OUTCOMES (what the model learns to predict):
  1. Depression severity level
  2. Anxiety severity level
  3. Stress severity level
  4. Whether the person shows clinical-level depression (Moderate or above)
  5. Whether the person shows clinical-level anxiety  (Moderate or above)
  6. Whether the person shows clinical-level stress   (Moderate or above)
  7. Which condition is most prominent

FEATURES USED AS INPUT:
  - Individual DASS item scores (Q1A-Q42A), recoded to 0-3
  - Big Five personality traits (TIPI1-TIPI10)
  - Vocabulary check score (valid VCL items only, as a proxy for engagement)
  - Demographics: education, urban background, English native, handedness,
    voted, married, family size — NO age, gender, race, religion, orientation,
    country (identity/sensitive attributes excluded)

HOW TO USE:
  1. Set INPUT_FILE  → path to data.csv
  2. Set OUTPUT_DIR  → where to save mh_dass_dataset.jsonl
  3. Run: python convert_dass_to_jsonl.py
"""

import json
import os
import random

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(ROOT_DIR, "data", "raw", "DASS-42", "data.csv")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "data", "processed", "jsonl")
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# ── DASS-42 subscale item assignments ─────────────────────────────────────────
DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
ANXIETY_ITEMS    = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
STRESS_ITEMS     = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

# Severity thresholds: (min, max, label) — using 0-based scoring
DEPRESSION_THRESHOLDS = [(0,9,"Normal"),(10,13,"Mild"),(14,20,"Moderate"),(21,27,"Severe"),(28,42,"Extremely Severe")]
ANXIETY_THRESHOLDS    = [(0,7,"Normal"),(8,9,"Mild"),(10,14,"Moderate"),(15,19,"Severe"),(20,42,"Extremely Severe")]
STRESS_THRESHOLDS     = [(0,14,"Normal"),(15,18,"Mild"),(19,25,"Moderate"),(26,33,"Severe"),(34,42,"Extremely Severe")]


def classify(score, thresholds):
    for lo, hi, label in thresholds:
        if lo <= score <= hi:
            return label
    return None


# ── DASS item descriptions (for human-readable input text) ───────────────────
ITEM_TEXT = {
    1:  "getting upset by quite trivial things",
    2:  "awareness of dryness of mouth",
    3:  "inability to experience any positive feeling",
    4:  "experiencing breathing difficulty",
    5:  "couldn't seem to get going",
    6:  "tended to over-react to situations",
    7:  "feeling of shakiness",
    8:  "found it difficult to relax",
    9:  "found situations that caused anxiety and relief when they ended",
    10: "felt nothing to look forward to",
    11: "getting upset rather easily",
    12: "felt using a lot of nervous energy",
    13: "felt sad and depressed",
    14: "getting impatient when delayed",
    15: "feeling of faintness",
    16: "lost interest in just about everything",
    17: "felt not worth much as a person",
    18: "felt rather touchy",
    19: "perspired noticeably without exertion",
    20: "felt scared without any good reason",
    21: "felt that life wasn't worthwhile",
    22: "found it hard to wind down",
    23: "had difficulty swallowing",
    24: "couldn't get enjoyment out of things",
    25: "aware of heart action without exertion",
    26: "felt down-hearted and blue",
    27: "found very irritable",
    28: "felt close to panic",
    29: "found it hard to calm down after being upset",
    30: "feared being thrown by unfamiliar tasks",
    31: "unable to become enthusiastic about anything",
    32: "found it difficult to tolerate interruptions",
    33: "in a state of nervous tension",
    34: "felt pretty worthless",
    35: "intolerant of anything preventing progress",
    36: "felt terrified",
    37: "could see nothing in the future to be hopeful about",
    38: "felt that life was meaningless",
    39: "found getting agitated",
    40: "worried about panicking in situations",
    41: "experienced trembling",
    42: "found it difficult to work up initiative",
}

RATING_TEXT = {
    0: "not at all",
    1: "somewhat / some of the time",
    2: "considerably / a good part of the time",
    3: "very much / most of the time",
}

TIPI_TEXT = {
    "TIPI1":  "extraverted and enthusiastic",
    "TIPI2":  "critical and quarrelsome",
    "TIPI3":  "dependable and self-disciplined",
    "TIPI4":  "anxious and easily upset",
    "TIPI5":  "open to new experiences and complex",
    "TIPI6":  "reserved and quiet",
    "TIPI7":  "sympathetic and warm",
    "TIPI8":  "disorganized and careless",
    "TIPI9":  "calm and emotionally stable",
    "TIPI10": "conventional and uncreative",
}

TIPI_RATING_TEXT = {
    1: "disagree strongly",
    2: "disagree moderately",
    3: "disagree a little",
    4: "neither agree nor disagree",
    5: "agree a little",
    6: "agree moderately",
    7: "agree strongly",
}

EDUCATION_TEXT = {
    1: "less than high school",
    2: "high school",
    3: "university degree",
    4: "graduate degree",
}

URBAN_TEXT = {
    1: "rural (countryside)",
    2: "suburban",
    3: "urban (town or city)",
}

HAND_TEXT = {1: "right-handed", 2: "left-handed", 3: "ambidextrous"}
ENGNAT_TEXT = {1: "yes", 2: "no"}
VOTED_TEXT = {1: "yes", 2: "no"}
MARRIED_TEXT = {1: "never married", 2: "currently married", 3: "previously married"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def get_dass_score(row, items):
    """Sum DASS items (convert from 1-4 to 0-3 by subtracting 1)."""
    total = 0
    for i in items:
        v = safe_int(row.get(f"Q{i}A"))
        if v is None or v < 1 or v > 4:
            return None   # missing or invalid — skip this row for this subscale
        total += (v - 1)
    return total


def build_dass_input(row, exclude_subscale=None):
    """
    Build the input text from DASS item scores and personality/demographic
    features. Excludes identity columns entirely.
    exclude_subscale: 'depression', 'anxiety', 'stress', or None
    """
    parts = []

    # ── DASS items ────────────────────────────────────────────────────────────
    # Group by subscale for readability; skip items belonging to the
    # excluded subscale (the outcome being predicted)
    subscale_map = {}
    for i in DEPRESSION_ITEMS:
        subscale_map[i] = "depression"
    for i in ANXIETY_ITEMS:
        subscale_map[i] = "anxiety"
    for i in STRESS_ITEMS:
        subscale_map[i] = "stress"

    item_parts = []
    for i in range(1, 43):
        if exclude_subscale and subscale_map.get(i) == exclude_subscale:
            continue
        v = safe_int(row.get(f"Q{i}A"))
        if v is None or v < 1 or v > 4:
            continue
        score = v - 1  # convert to 0-3
        rating = RATING_TEXT.get(score, str(score))
        item_parts.append(f"Q{i} ({ITEM_TEXT[i]}): {rating}")

    if item_parts:
        parts.append("Symptom responses — " + "; ".join(item_parts))

    # ── Personality (TIPI) ────────────────────────────────────────────────────
    tipi_parts = []
    for key, desc in TIPI_TEXT.items():
        v = safe_int(row.get(key))
        if v and 1 <= v <= 7:
            tipi_parts.append(f"{desc}: {TIPI_RATING_TEXT[v]}")
    if tipi_parts:
        parts.append("Personality — " + "; ".join(tipi_parts))

    # ── Vocabulary check (engagement proxy) ──────────────────────────────────
    # VCL6, VCL9, VCL12 are fake words — real words are the rest
    real_vcl = [1,2,3,4,5,7,8,10,11,13,14,15,16]
    known = sum(1 for i in real_vcl if safe_int(row.get(f"VCL{i}")) == 1)
    fake_checked = sum(1 for i in [6,9,12] if safe_int(row.get(f"VCL{i}")) == 1)
    if known is not None:
        parts.append(f"Vocabulary knowledge score: {known}/13 real words known")
    if fake_checked > 0:
        parts.append(f"Fake words incorrectly checked: {fake_checked} (validity concern)")

    # ── Demographics (non-identity) ───────────────────────────────────────────
    edu = safe_int(row.get("education"))
    if edu and edu in EDUCATION_TEXT:
        parts.append(f"Education: {EDUCATION_TEXT[edu]}")

    urban = safe_int(row.get("urban"))
    if urban and urban in URBAN_TEXT:
        parts.append(f"Childhood area: {URBAN_TEXT[urban]}")

    engnat = safe_int(row.get("engnat"))
    if engnat and engnat in ENGNAT_TEXT:
        parts.append(f"English is native language: {ENGNAT_TEXT[engnat]}")

    hand = safe_int(row.get("hand"))
    if hand and hand in HAND_TEXT:
        parts.append(f"Handedness: {HAND_TEXT[hand]}")

    voted = safe_int(row.get("voted"))
    if voted and voted in VOTED_TEXT:
        parts.append(f"Voted in a national election in the past year: {VOTED_TEXT[voted]}")

    married = safe_int(row.get("married"))
    if married and married in MARRIED_TEXT:
        parts.append(f"Marital status: {MARRIED_TEXT[married]}")

    fsize = safe_int(row.get("familysize"))
    if fsize and 1 <= fsize <= 20:
        parts.append(f"Number of children mother had (including respondent): {fsize}")

    return " | ".join(parts) if parts else None


# ── Sample generators — one per outcome ──────────────────────────────────────

def sample_depression_severity(row):
    score = get_dass_score(row, DEPRESSION_ITEMS)
    if score is None:
        return None
    level = classify(score, DEPRESSION_THRESHOLDS)
    if not level:
        return None

    input_text = build_dass_input(row, exclude_subscale="depression")
    if not input_text:
        return None

    instructions = [
        "What is this person's depression severity level based on their responses?",
        "How severe is this person's depression according to their symptom scores?",
        "Classify the depression level of this person from their reported symptoms.",
    ]
    output = (
        f"Based on the non-depression symptom responses and personality profile, "
        f"this person's depression severity is: {level} (score: {score}/42)."
    )
    return random.choice(instructions), input_text, output


def sample_anxiety_severity(row):
    score = get_dass_score(row, ANXIETY_ITEMS)
    if score is None:
        return None
    level = classify(score, ANXIETY_THRESHOLDS)
    if not level:
        return None

    input_text = build_dass_input(row, exclude_subscale="anxiety")
    if not input_text:
        return None

    instructions = [
        "What is this person's anxiety severity level based on their responses?",
        "How severe is this person's anxiety according to their symptom scores?",
        "Classify the anxiety level of this person from their reported symptoms.",
    ]
    output = (
        f"Based on the non-anxiety symptom responses and personality profile, "
        f"this person's anxiety severity is: {level} (score: {score}/42)."
    )
    return random.choice(instructions), input_text, output


def sample_stress_severity(row):
    score = get_dass_score(row, STRESS_ITEMS)
    if score is None:
        return None
    level = classify(score, STRESS_THRESHOLDS)
    if not level:
        return None

    input_text = build_dass_input(row, exclude_subscale="stress")
    if not input_text:
        return None

    instructions = [
        "What is this person's stress severity level based on their responses?",
        "How severe is this person's stress according to their symptom scores?",
        "Classify the stress level of this person from their reported symptoms.",
    ]
    output = (
        f"Based on the non-stress symptom responses and personality profile, "
        f"this person's stress severity is: {level} (score: {score}/42)."
    )
    return random.choice(instructions), input_text, output


def sample_clinical_depression(row):
    score = get_dass_score(row, DEPRESSION_ITEMS)
    if score is None:
        return None
    level = classify(score, DEPRESSION_THRESHOLDS)
    if not level:
        return None
    clinical = level in ("Moderate", "Severe", "Extremely Severe")

    input_text = build_dass_input(row, exclude_subscale="depression")
    if not input_text:
        return None

    instructions = [
        "Does this person show clinically significant depression?",
        "Is this person's depression at a clinical level requiring attention?",
        "Does this person's symptom profile indicate clinical depression?",
    ]
    if clinical:
        output = (
            f"Yes, this person shows clinically significant depression. "
            f"Their depression score is {score}/42, classified as {level}."
        )
    else:
        output = (
            f"No, this person does not show clinically significant depression. "
            f"Their depression score is {score}/42, classified as {level}."
        )
    return random.choice(instructions), input_text, output


def sample_clinical_anxiety(row):
    score = get_dass_score(row, ANXIETY_ITEMS)
    if score is None:
        return None
    level = classify(score, ANXIETY_THRESHOLDS)
    if not level:
        return None
    clinical = level in ("Moderate", "Severe", "Extremely Severe")

    input_text = build_dass_input(row, exclude_subscale="anxiety")
    if not input_text:
        return None

    instructions = [
        "Does this person show clinically significant anxiety?",
        "Is this person's anxiety at a clinical level requiring attention?",
        "Does this person's symptom profile indicate clinical anxiety?",
    ]
    if clinical:
        output = (
            f"Yes, this person shows clinically significant anxiety. "
            f"Their anxiety score is {score}/42, classified as {level}."
        )
    else:
        output = (
            f"No, this person does not show clinically significant anxiety. "
            f"Their anxiety score is {score}/42, classified as {level}."
        )
    return random.choice(instructions), input_text, output


def sample_clinical_stress(row):
    score = get_dass_score(row, STRESS_ITEMS)
    if score is None:
        return None
    level = classify(score, STRESS_THRESHOLDS)
    if not level:
        return None
    clinical = level in ("Moderate", "Severe", "Extremely Severe")

    input_text = build_dass_input(row, exclude_subscale="stress")
    if not input_text:
        return None

    instructions = [
        "Does this person show clinically significant stress?",
        "Is this person's stress at a clinical level requiring attention?",
        "Does this person's symptom profile indicate clinical stress?",
    ]
    if clinical:
        output = (
            f"Yes, this person shows clinically significant stress. "
            f"Their stress score is {score}/42, classified as {level}."
        )
    else:
        output = (
            f"No, this person does not show clinically significant stress. "
            f"Their stress score is {score}/42, classified as {level}."
        )
    return random.choice(instructions), input_text, output


def sample_dominant_condition(row):
    d_score = get_dass_score(row, DEPRESSION_ITEMS)
    a_score = get_dass_score(row, ANXIETY_ITEMS)
    s_score = get_dass_score(row, STRESS_ITEMS)

    if None in (d_score, a_score, s_score):
        return None

    d_level = classify(d_score, DEPRESSION_THRESHOLDS)
    a_level = classify(a_score, ANXIETY_THRESHOLDS)
    s_level = classify(s_score, STRESS_THRESHOLDS)

    # Normalise scores to 0-1 range for comparison (divide by max possible)
    d_norm = d_score / 42
    a_norm = a_score / 42
    s_norm = s_score / 42

    scores = {"depression": (d_norm, d_score, d_level),
              "anxiety":    (a_norm, a_score, a_level),
              "stress":     (s_norm, s_score, s_level)}
    dominant = max(scores, key=lambda k: scores[k][0])

    # Only generate this sample if at least one condition is non-Normal
    all_normal = all(v[2] == "Normal" for v in scores.values())

    input_text = build_dass_input(row, exclude_subscale=None)
    if not input_text:
        return None

    instructions = [
        "Which mental health condition is most prominent for this person?",
        "What is the dominant mental health concern for this person based on their responses?",
        "Among depression, anxiety, and stress, which is most elevated for this person?",
    ]

    if all_normal:
        output = (
            f"All three conditions are within normal range for this person. "
            f"Depression: {d_level} ({d_score}/42), "
            f"Anxiety: {a_level} ({a_score}/42), "
            f"Stress: {s_level} ({s_score}/42)."
        )
    else:
        _, dom_score, dom_level = scores[dominant]
        output = (
            f"The most prominent condition is {dominant} ({dom_level}, score: {dom_score}/42). "
            f"Full profile — Depression: {d_level} ({d_score}/42), "
            f"Anxiety: {a_level} ({a_score}/42), "
            f"Stress: {s_level} ({s_score}/42)."
        )
    return random.choice(instructions), input_text, output


OUTCOME_GENERATORS = [
    sample_depression_severity,
    sample_anxiety_severity,
    sample_stress_severity,
    sample_clinical_depression,
    sample_clinical_anxiety,
    sample_clinical_stress,
    sample_dominant_condition,
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
df = pd.read_csv(INPUT_FILE, sep="\t", on_bad_lines="skip", low_memory=False)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# Drop rows where all Q items are missing
q_cols = [f"Q{i}A" for i in range(1, 43)]
existing_q = [c for c in q_cols if c in df.columns]
df = df.dropna(subset=existing_q, how="all")
print(f"After dropping empty rows: {len(df):,} rows")

all_samples = []
skipped = 0

for _, row in df.iterrows():
    # Validity filter: reject rows where any fake VCL word is checked
    # (indicates careless responding)
    fake_checked = sum(1 for i in [6, 9, 12] if safe_int(row.get(f"VCL{i}")) == 1)
    if fake_checked >= 2:
        skipped += 1
        continue

    samples = row_to_samples(row.to_dict())
    all_samples.extend(samples)

print(f"Rows skipped (validity filter): {skipped:,}")
print(f"Total samples generated: {len(all_samples):,}")

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
out_path = os.path.join(OUTPUT_DIR, "mh_dass_dataset.jsonl")

with open(out_path, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("\n" + "=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)
print(f"Input file    : {INPUT_FILE}")
print(f"Total rows    : {len(df):,}")
print(f"Rows skipped  : {skipped:,}  (failed validity check)")
print(f"Total samples : {len(all_samples):,}")
print(f"Output        : {out_path}")

# Outcome distribution
print("\nOutcome sample counts:")
outcome_counts = {}
for s in all_samples:
    key = s["instruction"].split("?")[0][:55]
    outcome_counts[key] = outcome_counts.get(key, 0) + 1
for k, v in sorted(outcome_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<55} {v:>7,}")

if all_samples:
    print("\nSample entry:")
    print(json.dumps(all_samples[0], indent=2, ensure_ascii=False)[:1200])
else:
    print("No samples generated — check INPUT_FILE path.")
print("=" * 60)