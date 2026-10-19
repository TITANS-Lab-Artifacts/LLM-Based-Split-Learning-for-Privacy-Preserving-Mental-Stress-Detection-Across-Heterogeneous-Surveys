"""
prepare_tabular_datasets.py
-----------------------------
Splits and mixes the four mental health CSV/Excel datasets for
tabular model (FT-Transformer) training. Mirrors the JSONL script
structure exactly:

  Task 1  — Individual splits for each source dataset
              → train_{name}.csv  +  test_{name}.csv

  Task 2  — Moderate heterogeneous mix  (OSMI + Student Depression)
              → train_mod_hetero.csv  +  test_mod_hetero.csv

  Task 3  — Full heterogeneous mix  (all four datasets)
              → train_hetero.csv  +  test_hetero.csv

OSMI processing:
  All 9 annual files are merged into one dataframe before splitting.
  Columns are normalised (markup stripped, mapped to canonical names)
  and a `survey_year` column is added. Identity columns (age, gender,
  race, country, state, timestamps) are dropped.

Column alignment for mixing:
  When datasets are mixed, only columns present in ALL participating
  datasets are kept (inner join on column names). This ensures a clean,
  consistent feature matrix for the model.

HOW TO USE:
  1. Verify all file paths in the CONFIG section
  2. Adjust TEST_N_* sizes if needed
  3. Run:  python prepare_tabular_datasets.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSMI_DIR = os.path.join(ROOT_DIR, "data", "raw", "OSMI")
OSMI_FILES = [
    "2022_responses.csv",
    "2023_responses.csv",
    "Mental Health in Tech Survey 2014.xlsx",
    "mental-heath-in-tech-2016_20161114.csv",
    "OSMI 2019 Mental Health in Tech Survey Results - OSMI Mental Health in Tech Survey 2019.csv",
    "OSMI 2020 Mental Health in Tech Survey Results .csv",
    "OSMI 2021 Mental Health in Tech Survey Results .csv",
    "OSMI Mental Health in Tech Survey 2017.csv",
    "OSMI Mental Health in Tech Survey 2018.csv",
]

DASS_FILE    = os.path.join(ROOT_DIR, "data", "raw", "DASS-42", "data.csv")
STUDENT_FILE = os.path.join(ROOT_DIR, "data", "raw", "Student-Depression", "Student Depression Dataset.csv")
GENERAL_FILE = os.path.join(ROOT_DIR, "data", "raw", "Mental-Health-Dataset", "Mental Health Dataset.csv")

OUTPUT_DIR   = os.path.join(ROOT_DIR, "data", "processed", "tabular")

# Test set sizes (chosen proportional to dataset size)
TEST_N_OSMI    = 100    # OSMI total ~4,700
TEST_N_DASS    = 1000   # DASS likely 40,000+
TEST_N_STUDENT = 1000   # Student: 27,901
TEST_N_GENERAL = 2000   # General: 292,364

TEST_N_MOD_HETERO  = 500   # OSMI + Student mix test size
TEST_N_FULL_HETERO = 1000  # all-four mix test size

RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── OSMI column normalisation ─────────────────────────────────────────────────

def strip_markup(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"_+([^_]+)_+", r"\1", text)
    return text.replace("\xa0", " ").strip()


OSMI_COLUMN_MAP = {
    "Timestamp": "_timestamp", "Start Date (UTC)": "_timestamp", "#": "_row_id",
    "Age": "_age", "What is your age?": "_age",
    "Gender": "_gender", "What is your gender?": "_gender",
    "What is your race?": "_race",
    "Country": "_country_live", "What country do you live in?": "_country_live",
    "If you live in the United States, which state or territory do you live in?": "_state_live",
    "What US state or territory do you live in?": "_state_live",
    "What country do you work in?": "_country_work",
    "What US state or territory do you work in?": "_state_work",
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
    "Would you be willing to discuss a mental health issue with your coworkers?": "discuss_mh_coworkers",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?": "discuss_mh_coworkers",
    "Would you feel comfortable discussing a mental health issue with your coworkers?": "discuss_mh_coworkers",
    "Would you be willing to discuss a mental health issue with your direct supervisor(s)?": "discuss_mh_supervisor",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "discuss_mh_supervisor",
    "Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?": "discuss_mh_supervisor",
    "Do you feel that your employer takes mental health as seriously as physical health?": "employer_mh_vs_ph_parity",
    "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?": "observed_neg_consequences",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "observed_neg_consequences",
    "Would you bring up a mental health issue with a potential employer in an interview?": "mh_in_interview",
    "Would you bring up your mental health with a potential employer in an interview?": "mh_in_interview",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?": "mh_hurts_career",
    "Are you openly identified at work as a person with a mental health issue?": "openly_identified_mh",
    "Do you have a family history of mental illness?": "family_mh_history",
    "Have you sought treatment for a mental health condition?": "sought_treatment",
    "Have you ever sought treatment for a mental health issue from a mental health professional?": "sought_treatment",
    "Have you ever sought treatment for a mental health disorder from a mental health professional?": "sought_treatment",
    "If you have a mental health condition, do you feel that it interferes with your work?": "mh_interferes_work",
    "Do you currently have a mental health disorder?": "current_mh_disorder",
    "Have you had a mental health disorder in the past?": "past_mh_disorder",
    "Have you ever been diagnosed with a mental health disorder?": "ever_diagnosed",
    "Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?": "mh_coverage",
    "Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?": "mh_coverage",
    "Do you believe your productivity is ever affected by a mental health issue?": "productivity_affected",
    "Overall, how well do you think the tech industry supports employees with mental health issues?": "industry_mh_support",
}

# Columns to drop: admin, free-text, and identity/demographic
OSMI_DROP_PATTERNS = [
    r"^_timestamp$", r"^_row_id$", r"^_age$", r"^_gender$", r"^_race$",
    r"^_country", r"^_state",
    r"^Submit Date", r"^Network ID", r"^Tags$", r"^Response Type$",
    r"^Describe the", r"^How has it affected", r"^Briefly describe",
    r"^If there is anything else", r"^Would you be willing to talk to one of us",
    r"^Why or why not", r"^Other\.", r"^Any additional",
]

OSMI_FILENAME_YEAR = {
    "Mental Health in Tech Survey 2014":  2014,
    "mental-heath-in-tech-2016":          2016,
    "OSMI Mental Health in Tech Survey 2017": 2017,
    "OSMI Mental Health in Tech Survey 2018": 2018,
    "OSMI 2019 Mental Health in Tech Survey": 2019,
    "OSMI 2020 Mental Health in Tech Survey": 2020,
    "OSMI 2021 Mental Health in Tech Survey": 2021,
    "2022_responses": 2022,
    "2023_responses": 2023,
}

# Identity columns to drop from non-OSMI datasets
IDENTITY_COLS = {
    # Student Depression
    "id", "Gender", "Age", "City",
    # DASS
    "age", "gender", "race", "religion", "orientation", "country",
    # General MH
    "Timestamp", "Gender", "Country",
}

# DASS fake VCL words (validity filter)
DASS_FAKE_VCL = ["VCL6", "VCL9", "VCL12"]

# DASS subscale assignments
DASS_DEPRESSION = [3,5,10,13,16,17,21,24,26,31,34,37,38,42]
DASS_ANXIETY    = [2,4,7,9,15,19,20,23,25,28,30,36,40,41]
DASS_STRESS     = [1,6,8,11,12,14,18,22,27,29,32,33,35,39]

DASS_THRESHOLDS = {
    "depression": [(0,9,"Normal"),(10,13,"Mild"),(14,20,"Moderate"),(21,27,"Severe"),(28,42,"Extremely Severe")],
    "anxiety":    [(0,7,"Normal"),(8,9,"Mild"),(10,14,"Moderate"),(15,19,"Severe"),(20,42,"Extremely Severe")],
    "stress":     [(0,14,"Normal"),(15,18,"Mild"),(19,25,"Moderate"),(26,33,"Severe"),(34,42,"Extremely Severe")],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_dass(score, scale):
    for lo, hi, label in DASS_THRESHOLDS[scale]:
        if lo <= score <= hi:
            return label
    return None


def split_df(df, n_test, seed=RANDOM_SEED):
    """Shuffle and split into (train, test). Raises if too small."""
    if len(df) < n_test:
        raise ValueError(
            f"Dataset has {len(df)} rows but {n_test} test samples requested."
        )
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_shuffled.iloc[n_test:].reset_index(drop=True), \
           df_shuffled.iloc[:n_test].reset_index(drop=True)


def save(df, path):
    df.to_csv(path, index=False)
    print(f"    → {os.path.basename(path)}  ({len(df):,} rows × {len(df.columns)} cols)")


def report_split(name, train, test, out_train, out_test):
    print(f"\n  [{name}]  total={len(train)+len(test):,}  "
          f"train={len(train):,}  test={len(test):,}")
    save(train, out_train)
    save(test,  out_test)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_osmi():
    """Merge all 9 OSMI files into one normalised dataframe."""
    frames = []
    for fname in OSMI_FILES:
        path = os.path.join(OSMI_DIR, fname)
        if not os.path.exists(path):
            print(f"  [WARN] Not found, skipping: {fname}")
            continue

        df = pd.read_excel(path) if path.endswith(".xlsx") \
             else pd.read_csv(path, on_bad_lines="skip")

        # Detect year from filename
        year = None
        for key, y in OSMI_FILENAME_YEAR.items():
            if key in fname:
                year = y
                break
        if year is None:
            m = re.search(r"(20\d{2})", fname)
            year = int(m.group(1)) if m else 0

        # Strip markup from column names
        df.columns = [strip_markup(c) for c in df.columns]

        # Map to canonical names
        df = df.rename(columns=lambda c: OSMI_COLUMN_MAP.get(c, c))

        # Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Drop admin/identity columns
        drop = [c for c in df.columns
                if any(re.search(p, c, re.I) for p in OSMI_DROP_PATTERNS)]
        df = df.drop(columns=drop, errors="ignore")

        df.insert(0, "survey_year", year)
        df.insert(1, "dataset_source", "osmi")
        frames.append(df)
        print(f"    Loaded OSMI {year}: {len(df)} rows, {len(df.columns)} cols")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"  OSMI merged: {len(combined):,} rows, {len(combined.columns)} cols")
    return combined


def load_dass():
    """Load DASS-42, apply validity filter, compute subscale scores."""
    print(f"  Loading DASS: {DASS_FILE}")
    df = pd.read_csv(DASS_FILE, sep="\t", on_bad_lines="skip", low_memory=False)
    print(f"    Raw: {len(df):,} rows, {len(df.columns)} cols")

    # Validity filter: drop rows with 2+ fake VCL words checked
    def fake_count(row):
        return sum(1 for col in DASS_FAKE_VCL
                   if str(row.get(col, 0)) in ("1", "1.0"))

    mask = df.apply(fake_count, axis=1) < 2
    df = df[mask].reset_index(drop=True)
    print(f"    After validity filter: {len(df):,} rows")

    # Compute subscale scores (raw 1-4 → 0-3 scoring)
    def subscale_score(row, items):
        vals = []
        for i in items:
            col = f"Q{i}A"
            try:
                v = float(row.get(col, np.nan))
                if 1 <= v <= 4:
                    vals.append(v - 1)
            except (ValueError, TypeError):
                pass
        return sum(vals) if len(vals) == 14 else np.nan

    df["depression_score"]   = df.apply(lambda r: subscale_score(r, DASS_DEPRESSION), axis=1)
    df["anxiety_score"]      = df.apply(lambda r: subscale_score(r, DASS_ANXIETY), axis=1)
    df["stress_score"]       = df.apply(lambda r: subscale_score(r, DASS_STRESS), axis=1)
    df["depression_level"]   = df["depression_score"].apply(
        lambda s: classify_dass(int(s), "depression") if pd.notna(s) else np.nan)
    df["anxiety_level"]      = df["anxiety_score"].apply(
        lambda s: classify_dass(int(s), "anxiety") if pd.notna(s) else np.nan)
    df["stress_level"]       = df["stress_score"].apply(
        lambda s: classify_dass(int(s), "stress") if pd.notna(s) else np.nan)

    # Drop identity columns
    drop_identity = [c for c in df.columns
                     if c.lower() in {i.lower() for i in IDENTITY_COLS}]
    # Also drop raw timing columns (not useful for tabular model)
    drop_timing = [c for c in df.columns
                   if re.search(r"(elapse|[EI]\d+$)", c)]
    df = df.drop(columns=drop_identity + drop_timing, errors="ignore")

    df.insert(0, "dataset_source", "dass")
    print(f"    Final: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_student():
    """Load Student Depression dataset, drop identity columns."""
    print(f"  Loading Student: {STUDENT_FILE}")
    df = pd.read_csv(STUDENT_FILE)
    print(f"    Raw: {len(df):,} rows, {len(df.columns)} cols")

    drop = [c for c in df.columns
            if c in IDENTITY_COLS or c.lower() in {i.lower() for i in IDENTITY_COLS}]
    df = df.drop(columns=drop, errors="ignore")
    df.insert(0, "dataset_source", "student")
    print(f"    Final: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_general():
    """Load Mental Health General dataset, drop identity columns."""
    print(f"  Loading General MH: {GENERAL_FILE}")
    df = pd.read_csv(GENERAL_FILE)
    print(f"    Raw: {len(df):,} rows, {len(df.columns)} cols")

    drop = [c for c in df.columns
            if c in IDENTITY_COLS or c.lower() in {i.lower() for i in IDENTITY_COLS}]
    df = df.drop(columns=drop, errors="ignore")
    df.insert(0, "dataset_source", "general")
    print(f"    Final: {len(df):,} rows, {len(df.columns)} cols")
    return df


def union_join(dfs):
    """
    Merge dataframes using a UNION (outer) join — keep ALL columns from
    all dataframes, filling missing values with NaN where a dataset does
    not have that column. This preserves every feature for the model.

    dataset_source column is placed first so the FT-Transformer can use
    it as the prediction target for heterogeneous mixes.
    """
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    # Move dataset_source to front
    cols = ["dataset_source"] + [c for c in merged.columns if c != "dataset_source"]
    merged = merged[cols]
    print(f"    Union columns: {len(merged.columns)}  "
          f"(NaN where dataset lacks a column)")
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 62)
print("LOADING DATASETS")
print("=" * 62)

df_osmi    = load_osmi()
df_dass    = load_dass()
df_student = load_student()
df_general = load_general()

datasets = {
    "osmi":    df_osmi,
    "dass":    df_dass,
    "student": df_student,
    "general": df_general,
}

test_sizes = {
    "osmi":    TEST_N_OSMI,
    "dass":    TEST_N_DASS,
    "student": TEST_N_STUDENT,
    "general": TEST_N_GENERAL,
}

# ── Task 1: Individual splits ─────────────────────────────────────────────────

print("\n" + "=" * 62)
print("TASK 1 — Individual splits")
print("=" * 62)

splits = {}   # store train splits for reuse in mixing tasks
for name, df in datasets.items():
    train, test = split_df(df, test_sizes[name])
    splits[name] = {"train": train, "test": test, "full": df}

    report_split(
        name, train, test,
        os.path.join(OUTPUT_DIR, f"train_{name}.csv"),
        os.path.join(OUTPUT_DIR, f"test_{name}.csv"),
    )

# ── Task 2: Moderate heterogeneous mix (OSMI + Student) ──────────────────────

print("\n" + "=" * 62)
print("TASK 2 — Moderate heterogeneous mix: OSMI + Student")
print("=" * 62)

mod_mix = union_join([df_osmi, df_student])
print(f"  Total columns after union: {len(mod_mix.columns)}")
mod_train, mod_test = split_df(mod_mix, TEST_N_MOD_HETERO)

report_split(
    "OSMI + Student", mod_train, mod_test,
    os.path.join(OUTPUT_DIR, "train_mod_hetero.csv"),
    os.path.join(OUTPUT_DIR, "test_mod_hetero.csv"),
)

# ── Task 3: Full heterogeneous mix (all four) ─────────────────────────────────

print("\n" + "=" * 62)
print("TASK 3 — Full heterogeneous mix: all four datasets")
print("=" * 62)

full_mix = union_join([df_osmi, df_dass, df_student, df_general])
print(f"  Total columns after union: {len(full_mix.columns)}")
full_train, full_test = split_df(full_mix, TEST_N_FULL_HETERO)

report_split(
    "All four", full_train, full_test,
    os.path.join(OUTPUT_DIR, "train_hetero.csv"),
    os.path.join(OUTPUT_DIR, "test_hetero.csv"),
)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("OUTPUT SUMMARY")
print("=" * 62)
print(f"{'File':<40} {'Rows':>8}  {'Cols':>5}  {'Size':>8}")
print("-" * 62)
total_size = 0
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    total_size += os.path.getsize(fpath)
    df_tmp = pd.read_csv(fpath, nrows=1)
    n_rows = sum(1 for _ in open(fpath)) - 1   # fast line count
    print(f"  {fname:<40} {n_rows:>8,}  {len(df_tmp.columns):>5}  "
          f"{size_kb:>6.0f} KB")
print("-" * 62)
print(f"  {'Total':<50} {total_size/1024/1024:>6.1f} MB")
print(f"\nAll files saved to:\n  {OUTPUT_DIR}")
print("=" * 62)