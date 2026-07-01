
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge.

Interpretable, feature-based ranker for the "Senior AI Engineer — Founding Team" JD.

Design thesis (see README.md): the dataset is adversarial. Naive keyword/embedding
similarity is engineered to fail because of (a) keyword-stuffers — non-technical
profiles loaded with AI skills, and (b) honeypots — profiles that look ideal but are
internally impossible. We therefore reason structurally about *what the JD means*:

    final = consistency_gate
          * ( w_fit*role_fit + w_skill*skill_trust + w_exp*experience
            + w_loc*location + w_lex*lexical )
          * behavioral_modifier

- role_fit (title-class x career-evidence) is the decisive anti-stuffer signal:
  a "Marketing Manager" scores ~0 no matter how many AI skills are listed, while a
  plain-language "built a recommendation system at a product company" profile scores
  high even without buzzwords.
- consistency_gate is the DQ-critical honeypot kill-switch: it zeroes out profiles
  with internal impossibilities (duration_months > time elapsed, expert@0-months,
  salary min>max, systematic skill-duration overflow vs stated experience).
- behavioral_modifier is a bounded availability factor (response rate, recency,
  open-to-work) that modulates but never dominates fit.

Runtime: single streaming pass over candidates.jsonl(.gz). Pure CPU, no network,
no model weights. ~10-40s for 100K candidates, well under the 5min / 16GB budget.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import gzip
import heapq
import io
import json
import math
import re
from datetime import date

# --------------------------------------------------------------------------- #
# Reference date: anchored to the dataset (max last_active = 2026-05-27, today
# 2026-06-16). Constant for deterministic, reproducible scoring (no wall clock).
# --------------------------------------------------------------------------- #
REFERENCE_DATE = date(2026, 6, 16)

# Component weights (tuned on the offline silver-label eval harness — see eval/).
W_FIT, W_SKILL, W_EXP, W_LOC, W_LEX = 0.46, 0.18, 0.12, 0.09, 0.15

# Keep a small buffer above 100 so rounding-induced reordering can't drop a
# genuine top-100 candidate.
TOPK_BUFFER = 160

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #

# Title classes. Order matters: first match wins within a tier; we check the
# most specific / decisive patterns first.
IDEAL_TITLE_PATTERNS = [
    "recommendation system", "recommender", "search engineer", "search & ranking",
    "ranking engineer", "relevance engineer", "retrieval engineer", "personalization",
    "applied scientist", "applied ml engineer", "applied machine learning",
    "machine learning engineer", "ml engineer", "ai engineer", "nlp engineer",
]
RESEARCH_TITLE_PATTERNS = [
    "research engineer", "research scientist", "ai research", "ml research",
    "applied research",
]
VISION_SPEECH_TITLE_PATTERNS = [
    "computer vision", "cv engineer", "vision engineer", "speech", "robotics",
    "perception engineer",
]
ADJACENT_TITLE_PATTERNS = [
    "data scientist", "data engineer", "analytics engineer", "software engineer",
    "backend engineer", "ml platform", "mlops", "ai specialist", "research analyst",
]
GENERIC_SWE_TITLE_PATTERNS = [
    "full stack", "full-stack", "frontend", "front end", "front-end", "mobile developer",
    "java developer", ".net developer", "dotnet", "qa engineer", "test engineer",
    "devops", "cloud engineer", "software developer", "web developer", "android",
    "ios developer", "sdet", "site reliability", "sre",
]
NONTECH_TITLE_PATTERNS = [
    "hr ", "human resource", "recruiter", "talent acquisition", "marketing",
    "sales", "account executive", "accountant", "finance", "content writer",
    "copywriter", "graphic designer", "ux designer", "ui designer", "visual designer",
    "mechanical engineer", "civil engineer", "electrical engineer", "business analyst",
    "project manager", "program manager", "product manager", "operations manager",
    "customer support", "customer success", "support engineer", "consultant",
    "scrum master", "delivery manager", "teacher", "professor",
]

# Career-evidence phrase categories (searched in summary + headline + role
# descriptions). These detect the *substance* the JD cares about, independent of
# the skills section — this is what surfaces plain-language Tier-5s.
EV_RETRIEVAL = [
    "semantic search", "embedding-based", "embedding based", "embeddings", "retrieval",
    "vector search", "nearest neighbor", "nearest-neighbor", "ann ", "dense retrieval",
    "hybrid search", "bm25", "two-tower", "two tower",
]
EV_RANKING = [
    "ranking", "learning to rank", "learning-to-rank", "ltr", "re-rank", "rerank",
    "recommendation", "recommender", "relevance", "discovery feed", "personaliz",
    "matching system", "candidate ranking",
]
EV_VECTORDB = [
    "faiss", "pinecone", "milvus", "weaviate", "qdrant", "opensearch",
    "elasticsearch", "elastic search", "vespa",
]
EV_EVAL = [
    "ndcg", "mrr", "map@", "mean average precision", "a/b test", "ab test",
    "a/b-test", "offline-online", "offline/online", "offline to online",
    "relevance judgment", "relevance judgement", "click-through", "ctr",
    "evaluation framework", "offline metric",
]
EV_PRODUCTION = [
    "production", "shipped", "deployed", "real users", "at scale", "millions",
    "10m", "100k", "1m+", "latency", "throughput", "served", "serving",
    "online a/b", "live traffic",
]
EV_NLP = [
    "nlp", "natural language", "transformer", "bert", "sentence-transformer",
    "sentence transformer", "language model", "llm", "text classification",
    "named entity", "question answering",
]
EV_MLINFRA = [
    "sentence-transformers", "hugging face", "huggingface", "pytorch", "tensorflow",
    "xgboost", "lightgbm", "feature pipeline", "feature engineering", "mlflow",
    "kubeflow", "model serving", "inference",
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "mphasis", "ltimindtree",
    "deloitte", "ibm global", "dxc",
]

PRODUCT_INDUSTRIES = [
    "fintech", "e-commerce", "ecommerce", "food delivery", "saas", "ai/ml", "ai",
    "edtech", "gaming", "conversational ai", "voice ai", "healthtech", "adtech",
    "transportation", "insurance tech", "software", "social", "marketplace",
]
SERVICES_INDUSTRIES = [
    "it services", "consulting", "manufacturing", "conglomerate", "paper products",
    "bpo", "staffing",
]

# JD-relevant skills (canonical lowercase) -> evidence category for corroboration.
RELEVANT_SKILLS = {
    "embeddings": "retr", "sentence transformers": "retr", "faiss": "retr",
    "pinecone": "retr", "milvus": "retr", "weaviate": "retr", "qdrant": "retr",
    "opensearch": "retr", "elasticsearch": "retr", "vector search": "retr",
    "information retrieval": "retr", "semantic search": "retr",
    "learning to rank": "rank", "learning-to-rank": "rank", "ranking": "rank",
    "recommendation systems": "rank", "recommender systems": "rank",
    "recommendation": "rank",
    "nlp": "nlp", "natural language processing": "nlp", "transformers": "nlp",
    "hugging face transformers": "nlp", "bert": "nlp", "llms": "nlp",
    "large language models": "nlp", "fine-tuning llms": "nlp", "rag": "nlp",
    "machine learning": "ml", "deep learning": "ml", "pytorch": "ml",
    "tensorflow": "ml", "scikit-learn": "ml", "xgboost": "ml", "lightgbm": "ml",
    "feature engineering": "ml", "mlops": "ml", "mlflow": "ml",
    "learning to rank ": "rank", "python": "py",
}

# Tier-1 Indian hubs the JD explicitly welcomes.
TIER1_PRIME = ["noida", "pune"]                      # JD's own offices
TIER1_OTHER = ["hyderabad", "bangalore", "bengaluru", "mumbai", "delhi",
               "gurgaon", "gurugram", "chennai", "ncr"]

PROFICIENCY_W = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def parse_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        y, m, d = s[:10].split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def months_between(d1, d2):
    if not d1 or not d2:
        return None
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def any_in(text, patterns):
    return any(p in text for p in patterns)


def count_in(text, patterns):
    return sum(1 for p in patterns if p in text)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def sat(x, k):
    """Saturating ramp: 0 -> 0, large -> ~1."""
    return 1.0 - math.exp(-x / k)


# --------------------------------------------------------------------------- #
# Title classification
# --------------------------------------------------------------------------- #
def classify_title(title):
    t = " " + (title or "").lower().strip() + " "
    is_junior = any(j in t for j in ("junior", "jr.", "jr ", "associate", "intern", "trainee"))
    is_senior = any(s in t for s in ("senior", "staff", "lead", "principal", "sr."))
    # Most specific first.
    if any_in(t, RESEARCH_TITLE_PATTERNS):
        cls = "research"
    elif any_in(t, VISION_SPEECH_TITLE_PATTERNS):
        cls = "vision"
    elif any_in(t, IDEAL_TITLE_PATTERNS):
        cls = "ideal"
    elif any_in(t, NONTECH_TITLE_PATTERNS):
        cls = "nontech"
    elif any_in(t, ADJACENT_TITLE_PATTERNS):
        cls = "adjacent"
    elif any_in(t, GENERIC_SWE_TITLE_PATTERNS):
        cls = "generic"
    else:
        cls = "unknown"
    return cls, is_junior, is_senior


# --------------------------------------------------------------------------- #
# Career-evidence extraction
# --------------------------------------------------------------------------- #
def extract_evidence(cand):
    """Concatenate all free text and score the JD-relevant evidence categories."""
    parts = []
    prof = cand.get("profile", {})
    parts.append(prof.get("summary", ""))
    parts.append(prof.get("headline", ""))
    for role in cand.get("career_history", []):
        parts.append(role.get("description", ""))
        parts.append(role.get("title", ""))
    text = " ".join(parts).lower()

    ev = {
        "retr": count_in(text, EV_RETRIEVAL),
        "rank": count_in(text, EV_RANKING),
        "vdb": count_in(text, EV_VECTORDB),
        "eval": count_in(text, EV_EVAL),
        "prod": count_in(text, EV_PRODUCTION),
        "nlp": count_in(text, EV_NLP),
        "infra": count_in(text, EV_MLINFRA),
    }
    # Core IR/ranking substance the JD is built around.
    core = ev["retr"] + ev["rank"] + ev["vdb"]
    rigor = ev["eval"]
    prod = ev["prod"]
    # Saturating composite in 0..1: rewards genuine retrieval/ranking + eval + prod.
    score = (0.45 * sat(core, 2.0)
             + 0.20 * sat(rigor, 1.5)
             + 0.15 * sat(prod, 2.0)
             + 0.12 * sat(ev["nlp"], 2.0)
             + 0.08 * sat(ev["infra"], 2.0))
    ev["text"] = text
    ev["score"] = clamp(score, 0.0, 1.0)
    ev["has_nlp_ir"] = (ev["nlp"] + ev["retr"] + ev["rank"]) >= 1
    ev["has_production"] = prod >= 1
    return ev


# --------------------------------------------------------------------------- #
# role_fit: the spine (title-class x career-evidence)
# --------------------------------------------------------------------------- #
def role_fit_score(cand, ev):
    prof = cand.get("profile", {})
    cls, is_junior, is_senior = classify_title(prof.get("current_title", ""))
    e = ev["score"]

    # Best title ever held across career history (a current generic title can hide
    # a strong ML past, and vice-versa).
    best_cls = cls
    rank_order = {"nontech": 0, "unknown": 1, "generic": 2, "vision": 3,
                  "research": 3, "adjacent": 4, "ideal": 5}
    for role in cand.get("career_history", []):
        rc, _, _ = classify_title(role.get("title", ""))
        if rank_order.get(rc, 0) > rank_order.get(best_cls, 0):
            best_cls = rc

    if cls == "ideal":
        base = 0.86 + 0.14 * e
        if is_senior:
            base = min(1.0, base + 0.04)
    elif cls == "adjacent":
        base = 0.50 + 0.45 * e            # promotable to ~0.95 with real evidence
    elif cls == "generic":
        base = 0.20 + 0.45 * e            # needs strong retrieval/ranking evidence
        if best_cls == "ideal":
            base = max(base, 0.55 + 0.35 * e)
    elif cls == "research":
        # Pure research is a disqualifier; production + retrieval evidence rescues it.
        prod_factor = 0.55 + 0.45 * (1.0 if ev["has_production"] else 0.0)
        base = (0.40 + 0.45 * e) * prod_factor
    elif cls == "vision":
        nlp_factor = 1.0 if ev["has_nlp_ir"] else 0.55
        base = (0.28 + 0.45 * e) * nlp_factor
    elif cls == "unknown":
        base = 0.18 + 0.40 * e
        if best_cls == "ideal":
            base = max(base, 0.50 + 0.35 * e)
    else:  # nontech — keyword stuffers live here; stay near zero regardless of skills
        base = 0.03 + 0.06 * e
        # A genuine ML past under a now-nontech title is rare but possible.
        if best_cls == "ideal" and ev["score"] > 0.4:
            base = max(base, 0.35 + 0.25 * e)

    if is_junior:
        base *= 0.55

    # Negative career signals (multiplicative).
    base *= consulting_penalty(cand)
    base *= product_company_factor(cand)
    return clamp(base, 0.0, 1.0), cls, best_cls


def consulting_penalty(cand):
    companies = [(r.get("company", "") or "").lower() for r in cand.get("career_history", [])]
    cur = (cand.get("profile", {}).get("current_company", "") or "").lower()
    all_co = companies + [cur]
    if not all_co:
        return 1.0
    consult = [c for c in all_co if any(f in c for f in CONSULTING_FIRMS)]
    # Only a *penalty* if the entire career is consulting (JD: fine if prior product co).
    nonempty = [c for c in all_co if c]
    if nonempty and len(consult) == len(nonempty):
        return 0.7
    return 1.0


def product_company_factor(cand):
    prof = cand.get("profile", {})
    ind = (prof.get("current_industry", "") or "").lower()
    inds = [ind] + [(r.get("industry", "") or "").lower() for r in cand.get("career_history", [])]
    has_product = any(any(p in i for p in ("fintech", "e-commerce", "ecommerce",
                     "food delivery", "saas", "ai/ml", "edtech", "gaming",
                     "conversational ai", "voice ai", "healthtech", "adtech",
                     "marketplace", "social")) for i in inds)
    services_only = all((not i) or any(s in i for s in
                        ("it services", "consulting", "manufacturing",
                         "conglomerate", "paper products")) for i in inds)
    if has_product:
        return 1.0
    if services_only:
        return 0.85
    return 0.93


# --------------------------------------------------------------------------- #
# skill_trust: anti keyword-stuffing (corroborated, depth-weighted)
# --------------------------------------------------------------------------- #
def skill_trust_score(cand, ev):
    sig = cand.get("redrob_signals", {})
    assess = sig.get("skill_assessment_scores", {}) or {}
    assess_lc = {k.lower(): v for k, v in assess.items()}
    total = 0.0
    top = []
    for sk in cand.get("skills", []):
        name = (sk.get("name", "") or "").lower().strip()
        if name not in RELEVANT_SKILLS:
            continue
        prof_w = PROFICIENCY_W.get(sk.get("proficiency", ""), 0.3)
        dur = clamp((sk.get("duration_months", 0) or 0) / 24.0, 0.0, 1.0)
        end = clamp((sk.get("endorsements", 0) or 0) / 20.0, 0.0, 1.0)
        raw = prof_w * (0.45 + 0.35 * dur + 0.20 * end)
        # Corroboration: is this skill's category reflected in the career text?
        cat = RELEVANT_SKILLS[name]
        corro = 1.0
        if cat == "retr" and (ev["retr"] + ev["vdb"]) == 0:
            corro = 0.35
        elif cat == "rank" and ev["rank"] == 0:
            corro = 0.35
        elif cat == "nlp" and ev["nlp"] == 0:
            corro = 0.45
        elif cat in ("ml", "py") and ev["score"] < 0.05:
            corro = 0.6
        contrib = raw * corro
        # Assessment-score backing is hard evidence the skill is real.
        if name in assess_lc:
            contrib *= (0.85 + 0.4 * (assess_lc[name] / 100.0))
        total += contrib
        top.append((contrib, sk.get("name", ""), sk.get("proficiency", "")))
    top.sort(reverse=True)
    return clamp(total / 4.0, 0.0, 1.0), [t[1] for t in top[:3]]


# --------------------------------------------------------------------------- #
# experience
# --------------------------------------------------------------------------- #
def experience_score(cand, ev):
    yoe = cand.get("profile", {}).get("years_of_experience", 0) or 0
    if yoe < 2:
        s = 0.20
    elif yoe < 3:
        s = 0.20 + (yoe - 2) * 0.40
    elif yoe < 5:
        s = 0.60 + (yoe - 3) * 0.20
    elif yoe <= 9:
        s = 1.0
    elif yoe <= 13:
        s = 1.0 - (yoe - 9) * 0.05
    elif yoe <= 16:
        s = 0.80 - (yoe - 13) * (0.20 / 3.0)
    else:
        s = 0.55
    # Pre-LLM-era ML bonus: relevant ML/IR role that started before 2021.
    if ev["score"] > 0.2:
        for role in cand.get("career_history", []):
            d = parse_date(role.get("start_date"))
            if d and d.year <= 2020:
                s = min(1.0, s + 0.05)
                break
    return clamp(s, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# location
# --------------------------------------------------------------------------- #
def location_score(cand):
    prof = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    country = (prof.get("country", "") or "").lower()
    loc = (prof.get("location", "") or "").lower()
    relocate = bool(sig.get("willing_to_relocate", False))
    if "india" in country:
        if any(c in loc for c in TIER1_PRIME):
            return 1.0
        if any(c in loc for c in TIER1_OTHER):
            return 0.95
        return 0.80 if not relocate else 0.85
    # Outside India: no visa sponsorship — heavy down-weight, relocation helps a little.
    return 0.45 if relocate else 0.20


# --------------------------------------------------------------------------- #
# consistency_gate: honeypot / impossibility kill-switch (DQ-critical)
# --------------------------------------------------------------------------- #
def consistency_gate(cand):
    """Return (gate_multiplier, list_of_flags).

    HARD flags are near-impossible in legitimate profiles and zero the candidate.
    Thresholds were calibrated empirically against the full pool (see eval/ and the
    README "Honeypot detection" note) to maximise precision: each flag fires on only
    tens of candidates, all genuinely impossible, while sparing strong real profiles.

      - impossible_tenure  (~19): duration_months exceeds the wall-clock time that
        has elapsed since the role started  (e.g. duration_months=166 on a role
        started ~33 months ago)  [CAND_0007353/8960/10294]
      - expert_zero_months (~21): >=2 skills marked "expert" with 0 months used
        [CAND_0003582]
      - skill_duration_overflow (~8): >=3 skills each used 3+ YEARS longer than the
        candidate's entire stated experience  [CAND_0001610]

    Deliberately NOT used:
      - salary min>max : occurs for ~19% of the pool (synthetic noise, not a signal).
      - skill duration > experience at a tight (<3yr) slack : pervasive because the
        synthetic years_of_experience field is noisy; would kill legitimate juniors.
    """
    prof = cand.get("profile", {})
    yoe = prof.get("years_of_experience", 0) or 0
    flags = []

    # 1) duration_months exceeds wall-clock time available for the role.
    for role in cand.get("career_history", []):
        dm = role.get("duration_months", 0) or 0
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date")) or REFERENCE_DATE
        elapsed = months_between(start, end)
        if elapsed is not None and dm > elapsed + 6:
            flags.append("impossible_tenure")
            break

    # 2) expert-level proficiency with zero months of use (>=2 to avoid noise).
    expert_zero = sum(1 for sk in cand.get("skills", [])
                      if sk.get("proficiency") == "expert"
                      and (sk.get("duration_months", 0) or 0) == 0)
    if expert_zero >= 2:
        flags.append("expert_zero_months")

    # 3) Systematic skill-duration overflow: >=3 skills used 3+ years longer than
    #    the entire stated career. A 3yr slack tolerates pre-career/academic use and
    #    the noisy YoE field; >=3 such skills at once is a constructed impossibility.
    cap = yoe * 12 + 36
    overflow = sum(1 for sk in cand.get("skills", [])
                   if (sk.get("duration_months", 0) or 0) > cap)
    if overflow >= 3:
        flags.append("skill_duration_overflow")

    if flags:
        return 0.02, flags
    return 1.0, flags


# --------------------------------------------------------------------------- #
# behavioral_modifier: bounded availability factor [0.6, 1.1]
# --------------------------------------------------------------------------- #
def behavioral_modifier(cand):
    sig = cand.get("redrob_signals", {})
    resp = clamp(sig.get("recruiter_response_rate", 0.0) or 0.0, 0.0, 1.0)

    last = parse_date(sig.get("last_active_date"))
    days = (REFERENCE_DATE - last).days if last else 240
    recency = clamp(1.0 - (days - 60) / 180.0, 0.0, 1.0) if days > 60 else 1.0

    otw = 1.0 if sig.get("open_to_work_flag") else 0.4
    icr = clamp(sig.get("interview_completion_rate", 0.0) or 0.0, 0.0, 1.0)
    completeness = clamp((sig.get("profile_completeness_score", 0) or 0) / 100.0, 0.0, 1.0)
    saved = clamp((sig.get("saved_by_recruiters_30d", 0) or 0) / 10.0, 0.0, 1.0)
    verified = ((1.0 if sig.get("verified_email") else 0.0)
                + (1.0 if sig.get("verified_phone") else 0.0)) / 2.0
    art = sig.get("avg_response_time_hours", 240) or 240
    resp_time = clamp(1.0 - (art - 48) / 192.0, 0.0, 1.0) if art > 48 else 1.0
    gh = sig.get("github_activity_score", -1)
    gh_norm = 0.5 if (gh is None or gh < 0) else clamp(gh / 50.0, 0.0, 1.0)  # -1 = neutral

    b = (0.28 * resp + 0.22 * recency + 0.10 * otw + 0.10 * icr
         + 0.08 * completeness + 0.06 * saved + 0.06 * verified
         + 0.05 * resp_time + 0.05 * gh_norm)
    return clamp(0.6 + 0.5 * b, 0.6, 1.1), days


# --------------------------------------------------------------------------- #
# Full scoring of one candidate
# --------------------------------------------------------------------------- #
def score_candidate(cand):
    ev = extract_evidence(cand)
    fit, cls, best_cls = role_fit_score(cand, ev)
    skill, top_skills = skill_trust_score(cand, ev)
    exp = experience_score(cand, ev)
    loc = location_score(cand)
    lex = ev["score"]                       # lexical/evidence corroboration (minor)
    gate, flags = consistency_gate(cand)
    beh, inactive_days = behavioral_modifier(cand)

    core = W_FIT * fit + W_SKILL * skill + W_EXP * exp + W_LOC * loc + W_LEX * lex
    final = gate * core * beh

    card = {
        "id": cand.get("candidate_id", ""),
        "title": cand.get("profile", {}).get("current_title", ""),
        "company": cand.get("profile", {}).get("current_company", ""),
        "yoe": cand.get("profile", {}).get("years_of_experience", 0),
        "location": cand.get("profile", {}).get("location", ""),
        "country": cand.get("profile", {}).get("country", ""),
        "cls": cls, "best_cls": best_cls,
        "fit": fit, "skill": skill, "exp": exp, "loc": loc, "lex": lex,
        "beh": beh, "gate": gate, "flags": flags,
        "top_skills": top_skills, "ev": {k: ev[k] for k in
            ("retr", "rank", "vdb", "eval", "prod", "nlp", "has_production", "has_nlp_ir")},
        "inactive_days": inactive_days,
        "notice": cand.get("redrob_signals", {}).get("notice_period_days", None),
        "resp": cand.get("redrob_signals", {}).get("recruiter_response_rate", None),
        "open_to_work": cand.get("redrob_signals", {}).get("open_to_work_flag", None),
    }
    return final, card


# --------------------------------------------------------------------------- #
# Reasoning generation (deterministic, fact-grounded, rank-aware)
# --------------------------------------------------------------------------- #
def evidence_phrase(card):
    ev = card["ev"]
    if ev["retr"] or ev["vdb"]:
        return "built embedding/retrieval systems"
    if ev["rank"]:
        return "shipped ranking/recommendation systems"
    if ev["nlp"]:
        return "production NLP work"
    return "applied ML work"


def make_reasoning(card, rank):
    yoe = card["yoe"]
    title = card["title"]
    parts = []

    # Lead clause — fit-led, tone scaled to rank band.
    if rank <= 10:
        lead = f"{title} ({yoe:.0f} yrs) — {evidence_phrase(card)}"
    elif rank <= 50:
        lead = f"{title} with {yoe:.0f} yrs; {evidence_phrase(card)}"
    else:
        lead = f"{title}, {yoe:.0f} yrs"
    if card["company"]:
        lead += f" at {card['company']}"
    parts.append(lead)

    # Concrete corroborated skills.
    if card["top_skills"]:
        parts.append("strengths: " + ", ".join(card["top_skills"][:3]))

    # Evaluation-rigor / product signal when present (JD priorities).
    if card["ev"]["eval"]:
        parts.append("shows ranking-evaluation experience (A/B, offline metrics)")

    # Location relevance.
    loc = card["location"]
    if loc and "india" in (card["country"] or "").lower():
        if any(c in loc.lower() for c in TIER1_PRIME + TIER1_OTHER):
            parts.append(f"{loc.split(',')[0]}-based")
    elif card["country"]:
        parts.append(f"based in {card['country']} (relocation needed)")

    # Honest concerns — required for credibility and rank-consistency.
    concerns = []
    if card["cls"] == "research":
        concerns.append("research-leaning title; verify production depth")
    if card["cls"] == "vision" and not card["ev"]["has_nlp_ir"]:
        concerns.append("vision background, light NLP/IR")
    if card["cls"] in ("nontech",):
        concerns.append("non-engineering title; AI skills not corroborated by role history")
    if card["cls"] == "generic" and card["fit"] < 0.45:
        concerns.append("generic SWE background, limited retrieval/ranking evidence")
    if isinstance(card["notice"], (int, float)) and card["notice"] >= 90:
        concerns.append(f"{int(card['notice'])}d notice")
    if card["inactive_days"] is not None and card["inactive_days"] > 150:
        concerns.append(f"inactive ~{card['inactive_days']}d")
    if isinstance(card["resp"], (int, float)) and card["resp"] < 0.2:
        concerns.append(f"low recruiter response ({card['resp']:.2f})")
    if card["open_to_work"] is False:
        concerns.append("not flagged open-to-work")

    text = "; ".join(parts)
    if concerns:
        # Top ranks: at most one concern (they're strong); lower ranks surface more.
        k = 1 if rank <= 10 else (2 if rank <= 50 else 3)
        text += ". Concerns: " + ", ".join(concerns[:k])
    text += "."
    # Keep to ~2 sentences / reasonable length.
    return text[:300]


# --------------------------------------------------------------------------- #
# IO + main
# --------------------------------------------------------------------------- #
def open_candidates(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl(.gz)")
    ap.add_argument("--out", required=True, help="output submission CSV path")
    ap.add_argument("--topn", type=int, default=100, help="number of rows to emit")
    args = ap.parse_args()

    # Min-heap of (score, id, card); keep TOPK_BUFFER best. id breaks ties so the
    # heap never needs to compare the card dicts.
    heap = []
    n = 0
    with open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            score, card = score_candidate(cand)
            item = (score, card["id"], card)
            if len(heap) < TOPK_BUFFER:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    # Final ordering: score desc, then candidate_id asc (validator tie-break rule).
    ordered = sorted(heap, key=lambda x: (-x[0], x[1]))
    top = ordered[: args.topn]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (score, cid, card) in enumerate(top):
            rank = i + 1
            reasoning = make_reasoning(card, rank)
            w.writerow([cid, rank, f"{score:.8f}", reasoning])

    print(f"Scored {n} candidates; wrote top {len(top)} to {args.out}")


if __name__ == "__main__":
    main()