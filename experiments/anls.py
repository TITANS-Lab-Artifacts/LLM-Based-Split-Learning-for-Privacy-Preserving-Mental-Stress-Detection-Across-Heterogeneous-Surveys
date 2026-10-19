"""
anls.py
=======
ANLS scoring with the SAME normalization used by the fine-tuning scripts.
Do NOT change the normalization — results must stay comparable across all runs.
"""

import re
import unicodedata


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


def nls(pred_core, gt_core, threshold=0.5):
    p = canonical(normalize_text(pred_core))
    g = canonical(normalize_text(gt_core))
    if not p and not g: return 1.0
    if not p or  not g: return 0.0
    if p == g:          return 1.0
    partial = severity_partial_credit(p, g)
    if partial > 0:     return partial
    sim = 1.0 - levenshtein(p, g) / max(len(p), len(g))
    return sim if sim >= threshold else 0.0


def compute_anls(pred, gt, threshold=0.5):
    pc, gc = extract_core_answer(pred), extract_core_answer(gt)
    return nls(pc, gc, threshold), pc, gc
