"""
convert_to_jsonl.py
--------------------
Converts OSMI Mental Health in Tech survey files into a single JSONL
file for LLaMA fine-tuning.

FORMAT (correct for supervised fine-tuning):
  instruction — a short, generic question about one outcome
  input       — the respondent's attribute values (the features)
  output      — the answer to the instruction based on those attributes

Example:
  instruction: "Did this tech worker seek mental health treatment?"
  input:       "Company size: 26-100. Employer provides MH benefits: No.
                Family history of mental illness: Yes. MH discussed at
                work: No. Medical leave ease: Somewhat difficult."
  output:      "Yes, this tech worker has sought professional mental
                health treatment."

Each row generates one sample per available outcome. Demographic and
identity columns (age, gender, race, country, state) are excluded from
both input and output.

Output: mh_tech_dataset.jsonl

HOW TO USE:
  1. Set INPUT_DIR  → folder with survey files
  2. Set OUTPUT_DIR → where to save the JSONL
  3. Run: python convert_to_jsonl.py
"""

import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(ROOT_DIR, "data", "raw", "OSMI")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed", "jsonl")
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)


# ── Column name cleaning ──────────────────────────────────────────────────────

def strip_markup(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"_+([^_]+)_+", r"\1", text)
    return text.replace("\xa0", " ").strip()


COLUMN_MAP = {
    "Timestamp": "timestamp", "Start Date (UTC)": "timestamp", "#": "_row_id",
    "Age": "age", "What is your age?": "age",
    "Gender": "gender", "What is your gender?": "gender",
    "What is your race?": "race",
    "Country": "country_live", "What country do you live in?": "country_live",
    "If you live in the United States, which state or territory do you live in?": "state_live",
    "What US state or territory do you live in?": "state_live",
    "What country do you work in?": "country_work",
    "What US state or territory do you work in?": "state_work",
    "Are you self-employed?": "self_employed",
    "How many employees does your company or organization have?": "company_size",
    "Is your employer primarily a tech company/organization?": "tech_employer",
    "Is your primary role within your company related to tech/IT?": "tech_role",
    "Do you work remotely (outside of an office) at least 50% of the time?": "remote_work",
    "Do you work remotely?": "remote_work",
    "Which of the following best describes your work position?": "work_position",
    "Does your employer provide mental health benefits?": "employer_mh_benefits",
    "Does your employer provide mental health benefits as part of healthcare coverage?": "employer_mh_benefits",
    "Do you know the options for mental health care your employer provides?": "employer_mh_options_known",
    "Do you know the options for mental health care available under your employer-provided coverage?": "employer_mh_options_known",
    "Do you know the options for mental health care available under your employer-provided health coverage?": "employer_mh_options_known",
    "Has your employer ever discussed mental health as part of an employee wellness program?": "employer_mh_discussed",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?": "employer_mh_discussed",
    "Does your employer provide resources to learn more about mental health issues and how to seek help?": "employer_mh_resources",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?": "employer_mh_resources",
    "Does your employer offer resources to learn more about mental health disorders and options for seeking help?": "employer_mh_resources",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?": "anonymity_protected",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "anonymity_protected",
    "How easy is it for you to take medical leave for a mental health condition?": "medical_leave_ease",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": "medical_leave_ease",
    "If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?": "medical_leave_ease",
    "Overall, how much importance does your employer place on mental health?": "employer_importance_mh",
    "Overall, how much importance does your employer place on physical health?": "employer_importance_ph",
    "Do you think that discussing a mental health issue with your employer would have negative consequences?": "neg_consequences_employer",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?": "neg_consequences_employer",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": "neg_consequences_employer_physical",
    "Would you be willing to discuss a mental health issue with your coworkers?": "discuss_mh_coworkers",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?": "discuss_mh_coworkers",
    "Would you feel comfortable discussing a mental health issue with your coworkers?": "discuss_mh_coworkers",
    "Would you be willing to discuss a mental health issue with your direct supervisor(s)?": "discuss_mh_supervisor",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "discuss_mh_supervisor",
    "Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?": "discuss_mh_supervisor",
    "Would you feel more comfortable talking to your coworkers about your physical health or your mental health?": "comfort_physical_vs_mental",
    "Have you ever discussed your mental health with your employer?": "discussed_mh_employer",
    "Have you ever discussed your mental health with coworkers?": "discussed_mh_coworkers_ever",
    "Have you ever had a coworker discuss their or another coworker's mental health with you?": "coworker_discussed_mh_with_you",
    "Do you feel that your employer takes mental health as seriously as physical health?": "employer_mh_vs_ph_parity",
    "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?": "observed_neg_consequences",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "observed_neg_consequences",
    "Would you bring up a mental health issue with a potential employer in an interview?": "mh_in_interview",
    "Would you bring up your mental health with a potential employer in an interview?": "mh_in_interview",
    "Would you bring up a physical health issue with a potential employer in an interview?": "ph_in_interview",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?": "ph_in_interview",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?": "mh_hurts_career",
    "Are you openly identified at work as a person with a mental health issue?": "openly_identified_mh",
    "Has being identified as a person with a mental health issue affected your career?": "mh_affected_career",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?": "coworkers_view_negatively",
    "If they knew you suffered from a mental health disorder, how do you think that team members/co-workers would react?": "coworkers_view_negatively",
    "If they knew you suffered from a mental health disorder, how do you think that your team members/co-workers would react?": "coworkers_view_negatively",
    "How willing would you be to share with friends and family that you have a mental illness?": "share_mh_friends_family",
    "Do you have a family history of mental illness?": "family_mh_history",
    "Have you sought treatment for a mental health condition?": "sought_treatment",
    "Have you ever sought treatment for a mental health issue from a mental health professional?": "sought_treatment",
    "Have you ever sought treatment for a mental health disorder from a mental health professional?": "sought_treatment",
    "If you have a mental health condition, do you feel that it interferes with your work?": "mh_interferes_work",
    "Do you currently have a mental health disorder?": "current_mh_disorder",
    "Have you had a mental health disorder in the past?": "past_mh_disorder",
    "Have you ever been diagnosed with a mental health disorder?": "ever_diagnosed",
    "Have been diagnosed with a mental health condition by a medical professional?": "ever_diagnosed",
    "Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?": "mh_coverage",
    "Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?": "mh_coverage",
    "If you have a mental health disorder, how often do you feel that it interferes with your work when being treated effectively?": "mh_interferes_treated",
    "If you have a mental health disorder, how often do you feel that it interferes with your work when NOT being treated effectively (i.e., when you are experiencing symptoms)?": "mh_interferes_untreated",
    "Do you believe your productivity is ever affected by a mental health issue?": "productivity_affected",
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?": "reveal_mh_clients",
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?": "reveal_mh_coworkers",
    "Do you know local or online resources to seek help for a mental health issue?": "knows_mh_resources",
    "Do you know local or online resources to seek help for a mental health disorder?": "knows_mh_resources",
    "Overall, how well do you think the tech industry supports employees with mental health issues?": "industry_mh_support",
    "Do you have previous employers?": "has_prev_employers",
    "Have your previous employers provided mental health benefits?": "prev_employer_mh_benefits",
    "Were you aware of the options for mental health care provided by your previous employers?": "prev_employer_mh_options_known",
    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?": "prev_employer_mh_discussed",
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?": "prev_employer_mh_resources",
    "Did your previous employers provide resources to learn more about mental health disorders and how to seek help?": "prev_employer_mh_resources",
    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?": "prev_anonymity_protected",
    "Did you feel that your previous employers took mental health as seriously as physical health?": "prev_employer_mh_vs_ph_parity",
    "Did you ever discuss your mental health with your previous employer?": "prev_discussed_mh_employer",
    "Overall, how much importance did your previous employer place on mental health?": "prev_employer_importance_mh",
    "Overall, how much importance did your previous employer place on physical health?": "prev_employer_importance_ph",
    "Have you been diagnosed with COVID-19?": "covid_diagnosed",
}

DROP_PATTERNS = [
    r"^Submit Date", r"^Network ID", r"^Tags$", r"^Response Type$", r"^_row_id$",
    r"^timestamp$", r"^Describe the", r"^How has it affected", r"^Briefly describe",
    r"^If there is anything else", r"^Would you be willing to talk to one of us",
    r"^Why or why not", r"^Other\.", r"^Any additional",
]

# Columns that are outcomes — excluded from input features
OUTCOME_COLS = {
    "sought_treatment", "current_mh_disorder", "mh_interferes_work",
    "productivity_affected", "industry_mh_support", "mh_hurts_career",
    "mh_affected_career", "openly_identified_mh",
}

# Columns that are identity/demographic — excluded entirely
IDENTITY_COLS = {
    "age", "gender", "race", "country_live", "state_live",
    "country_work", "state_work", "survey_year",
}

FILENAME_YEAR = {
    "Mental_Health_in_Tech_Survey_2014": 2014,
    "Mental Health in Tech Survey 2014":  2014,
    "mental-heath-in-tech-2016":          2016,
    "OSMI_Mental_Health_in_Tech_Survey_2017": 2017,
    "OSMI Mental Health in Tech Survey 2017": 2017,
    "OSMI_Mental_Health_in_Tech_Survey_2018": 2018,
    "OSMI Mental Health in Tech Survey 2018": 2018,
    "OSMI_2019_Mental_Health_in_Tech_Survey": 2019,
    "OSMI 2019 Mental Health in Tech Survey": 2019,
    "OSMI_2020_Mental_Health_in_Tech_Survey": 2020,
    "OSMI 2020 Mental Health in Tech Survey": 2020,
    "OSMI_2021_Mental_Health_in_Tech_Survey": 2021,
    "OSMI 2021 Mental Health in Tech Survey": 2021,
    "2022_responses": 2022,
    "2023_responses": 2023,
}

# Human-readable labels for feature columns used in input
FEATURE_LABELS = {
    "self_employed":              "Self-employed",
    "company_size":               "Company size",
    "tech_employer":              "Employer is a tech company",
    "tech_role":                  "Role is tech/IT-related",
    "remote_work":                "Remote work",
    "work_position":              "Job position",
    "employer_mh_benefits":       "Employer provides mental health benefits",
    "employer_mh_options_known":  "Aware of employer mental health care options",
    "employer_mh_discussed":      "Mental health formally discussed at work",
    "employer_mh_resources":      "Employer provides mental health resources",
    "anonymity_protected":        "Anonymity protected when using treatment resources",
    "medical_leave_ease":         "Ease of taking medical leave for mental health",
    "employer_importance_mh":     "Employer importance rating for mental health (0-10)",
    "employer_importance_ph":     "Employer importance rating for physical health (0-10)",
    "employer_mh_vs_ph_parity":   "Employer treats mental health as seriously as physical",
    "neg_consequences_employer":  "Discussing MH with employer would have negative consequences",
    "discuss_mh_coworkers":       "Comfortable discussing MH with coworkers",
    "discuss_mh_supervisor":      "Comfortable discussing MH with supervisor",
    "discussed_mh_employer":      "Has discussed MH with employer",
    "discussed_mh_coworkers_ever":"Has discussed MH with coworkers",
    "coworker_discussed_mh_with_you": "A coworker has discussed MH with them",
    "observed_neg_consequences":  "Observed negative consequences for coworkers who disclosed MH",
    "comfort_physical_vs_mental": "More comfortable discussing physical or mental health at work",
    "family_mh_history":          "Family history of mental illness",
    "mh_coverage":                "Has medical coverage for mental health treatment",
    "mh_interferes_treated":      "MH interference with work when treated",
    "mh_interferes_untreated":    "MH interference with work when not treated",
    "reveal_mh_clients":          "Reveals MH diagnosis to clients",
    "reveal_mh_coworkers":        "Reveals MH diagnosis to coworkers",
    "knows_mh_resources":         "Knows local/online mental health resources",
    "coworkers_view_negatively":  "Believes coworkers would view them negatively",
    "share_mh_friends_family":    "Willingness to share MH with friends/family",
    "mh_in_interview":            "Would bring up MH in a job interview",
    "ph_in_interview":            "Would bring up physical health in a job interview",
    "past_mh_disorder":           "Past mental health disorder",
    "ever_diagnosed":             "Ever diagnosed with a mental health disorder",
    "has_prev_employers":         "Has previous employers",
    "prev_employer_mh_benefits":  "Previous employers provided MH benefits",
    "prev_employer_mh_options_known": "Aware of previous employer MH care options",
    "prev_employer_mh_discussed": "MH formally discussed at previous workplaces",
    "prev_employer_mh_resources": "Previous employers provided MH resources",
    "prev_anonymity_protected":   "Anonymity protected at previous employers",
    "prev_employer_mh_vs_ph_parity": "Previous employer treated MH as seriously as physical",
    "prev_discussed_mh_employer": "Discussed MH with previous employer",
    "prev_employer_importance_mh":"Previous employer importance rating for MH (0-10)",
    "prev_employer_importance_ph":"Previous employer importance rating for physical health (0-10)",
}


# ── File loading ──────────────────────────────────────────────────────────────

def detect_year(path):
    name = os.path.basename(path)
    for key, year in FILENAME_YEAR.items():
        if key in name:
            return year
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None


def load_file(path, year):
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path, on_bad_lines="skip")
    df.columns = [strip_markup(c) for c in df.columns]
    df = df.rename(columns=lambda c: COLUMN_MAP.get(c, c))
    df = df.loc[:, ~df.columns.duplicated()]
    drop_cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in DROP_PATTERNS)]
    df = df.drop(columns=drop_cols, errors="ignore")
    df.insert(0, "survey_year", year)
    return df


# ── Value helpers ─────────────────────────────────────────────────────────────

def clean_val(v):
    """Return string value or None if missing/invalid."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "n/a", "-") else None


def yn(v):
    """Normalise 1/0/True/False/Yes/No to readable form."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "1.0", "yes", "true"):
        return "Yes"
    if s in ("0", "0.0", "no", "false"):
        return "No"
    return str(v).strip()


def build_input_text(row, exclude_cols):
    """
    Build the input string from all available feature columns,
    excluding outcome and identity columns.
    """
    parts = []
    for col, label in FEATURE_LABELS.items():
        if col in exclude_cols:
            continue
        raw = clean_val(row.get(col))
        if raw is None:
            continue
        value = yn(raw)
        parts.append(f"{label}: {value}")
    return ". ".join(parts) + "." if parts else None


# ── Sample generators — one per outcome ──────────────────────────────────────
# Each takes a row dict and the pre-built input_text string.
# Returns (instruction, input_text, output) or None if outcome is missing.

def sample_sought_treatment(row, input_text):
    v = yn(clean_val(row.get("sought_treatment")))
    if not v:
        return None
    instructions = [
        "Did this tech worker seek professional mental health treatment?",
        "Has this tech worker ever sought help from a mental health professional?",
        "Did this person get professional treatment for mental health?",
    ]
    if v == "Yes":
        output = "Yes, this tech worker has sought professional mental health treatment."
    else:
        output = "No, this tech worker has not sought professional mental health treatment."
    return random.choice(instructions), input_text, output


def sample_current_mh_disorder(row, input_text):
    v = clean_val(row.get("current_mh_disorder"))
    if not v:
        return None
    v = yn(v)
    instructions = [
        "Does this tech worker currently have a mental health disorder?",
        "What is this tech worker's current mental health disorder status?",
        "Is this person currently experiencing a mental health disorder?",
    ]
    outputs = {
        "Yes":     "Yes, this tech worker currently has a mental health disorder.",
        "No":      "No, this tech worker does not currently have a mental health disorder.",
        "Maybe":   "This tech worker is unsure whether they currently have a mental health disorder.",
        "Possibly":"This tech worker thinks they possibly have a current mental health disorder.",
    }
    output = outputs.get(v, f"This tech worker's current mental health disorder status is: {v}.")
    return random.choice(instructions), input_text, output


def sample_mh_interferes_work(row, input_text):
    v = clean_val(row.get("mh_interferes_work"))
    if not v:
        return None
    instructions = [
        "How often does mental health interfere with this tech worker's work?",
        "Does mental health affect this person's ability to work?",
        "How much does mental health impact this tech worker's job performance?",
    ]
    output = f"Mental health '{v}' interferes with this tech worker's ability to perform their job."
    return random.choice(instructions), input_text, output


def sample_productivity_affected(row, input_text):
    v = clean_val(row.get("productivity_affected"))
    if not v:
        return None
    instructions = [
        "Is this tech worker's productivity affected by mental health?",
        "Does mental health impact this person's work productivity?",
        "How does mental health affect this tech worker's productivity?",
    ]
    v_norm = yn(v)
    if v_norm == "Yes":
        output = "Yes, this tech worker's productivity is affected by mental health issues."
    elif v_norm == "No":
        output = "No, this tech worker's productivity is not affected by mental health issues."
    else:
        output = f"Regarding productivity impact from mental health: '{v}'."
    return random.choice(instructions), input_text, output


def sample_industry_mh_support(row, input_text):
    v = clean_val(row.get("industry_mh_support"))
    if not v:
        return None
    try:
        score = float(v)
        level = "poor" if score <= 2 else ("moderate" if score == 3 else "good")
        output = (f"This tech worker rates the tech industry's mental health support "
                  f"as {v}/5, which is considered {level}.")
    except ValueError:
        output = f"This tech worker's view of mental health support in the tech industry: '{v}'."
    instructions = [
        "How does this tech worker rate mental health support in the tech industry?",
        "What is this person's view of how the tech industry handles mental health?",
        "How well does this tech worker think the tech industry supports mental health?",
    ]
    return random.choice(instructions), input_text, output


def sample_mh_hurts_career(row, input_text):
    v = clean_val(row.get("mh_hurts_career"))
    if not v:
        return None
    instructions = [
        "Does this tech worker think mental health issues would hurt their career?",
        "Would being identified with a mental health issue affect this person's career?",
        "How does this tech worker feel about mental health impacting career prospects?",
    ]
    output = (f"This tech worker's view on whether mental health would hurt their career: '{v}'.")
    return random.choice(instructions), input_text, output


def sample_openly_identified(row, input_text):
    v = yn(clean_val(row.get("openly_identified_mh")))
    if not v:
        return None
    instructions = [
        "Is this tech worker openly identified at work as having a mental health issue?",
        "Does this person openly disclose their mental health condition at work?",
        "Is this tech worker open about their mental health in the workplace?",
    ]
    if v == "Yes":
        output = "Yes, this tech worker is openly identified at work as having a mental health issue."
    elif v == "No":
        output = "No, this tech worker is not openly identified at work as having a mental health issue."
    else:
        output = f"This tech worker's openness about mental health at work: '{v}'."
    return random.choice(instructions), input_text, output


OUTCOME_GENERATORS = [
    ("sought_treatment",      sample_sought_treatment),
    ("current_mh_disorder",   sample_current_mh_disorder),
    ("mh_interferes_work",    sample_mh_interferes_work),
    ("productivity_affected", sample_productivity_affected),
    ("industry_mh_support",   sample_industry_mh_support),
    ("mh_hurts_career",       sample_mh_hurts_career),
    ("openly_identified_mh",  sample_openly_identified),
]


def row_to_samples(row):
    samples = []
    for outcome_col, gen_fn in OUTCOME_GENERATORS:
        # Build input text excluding this specific outcome column
        exclude = OUTCOME_COLS | IDENTITY_COLS | {outcome_col}
        input_text = build_input_text(row, exclude)
        if not input_text:
            continue
        result = gen_fn(row, input_text)
        if result:
            instruction, inp, output = result
            samples.append({
                "instruction": instruction,
                "input":       inp,
                "output":      output,
            })
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

all_files = (
    glob.glob(os.path.join(INPUT_DIR, "*.csv")) +
    glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))
)

all_samples = []
file_stats  = []

for path in sorted(all_files):
    year = detect_year(path)
    if not year:
        print(f"[SKIP] Cannot determine year: {os.path.basename(path)}")
        continue

    df = load_file(path, year)
    file_samples = []
    for _, row in df.iterrows():
        file_samples.extend(row_to_samples(row.to_dict()))

    all_samples.extend(file_samples)
    file_stats.append((year, os.path.basename(path), len(df), len(file_samples)))
    print(f"  {year}: {len(df):>5} rows → {len(file_samples):>5} samples  ({os.path.basename(path)})")

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
out_path = os.path.join(OUTPUT_DIR, "mh_tech_dataset.jsonl")

with open(out_path, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("\n" + "=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)
for year, fname, rows, samples in sorted(file_stats):
    print(f"  {year}  {fname[:48]:<50}  {rows:>5} rows  {samples:>6} samples")
print(f"  {'TOTAL':<55}  {len(all_samples):>6} samples")
print(f"\nOutput : {out_path}")
if all_samples:
    print("\nSample entry:")
    print(json.dumps(all_samples[0], indent=2, ensure_ascii=False))
else:
    print("No samples generated — check that INPUT_DIR contains the survey files.")
print("=" * 60)