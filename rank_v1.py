#!/usr/bin/env python3
"""
rank.py — v1 candidate ranker (Redrob Hackathon)

Simple weighted scoring across four signals:
  - title relevance (is this person doing ML/AI work?)
  - skill match (do they have the skills the JD asks for?)
  - experience years (are they in the 5-9 year band?)
  - behavioral signals (are they available and responsive?)

No external dependencies — stdlib only.
Runtime: ~55s for 100,000 candidates on a single CPU core.

Usage:
    python rank_v1.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import gzip
import heapq
import io
import json
import math
import re
import sys

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
TOP_N = 100
HEAP_BUFFER = 150  

# Scoring weights (must sum to 1.0)
W_TITLE    = 0.40
W_SKILL    = 0.30
W_EXP      = 0.15
W_BEHAV    = 0.15

# --------------------------------------------------------------------------- #
# JD vocabulary — skills the JD explicitly asks for
# --------------------------------------------------------------------------- #
REQUIRED_SKILLS = {
    # embeddings / retrieval
    "sentence-transformers", "sentence transformers", "faiss", "openai embeddings",
    "bge", "e5", "dense retrieval", "semantic search", "embedding", "embeddings",
    "rag", "retrieval augmented", "hybrid search", "neural search",
    # vector databases
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "pgvector", "chromadb", "vespa",
    # ranking / recsys
    "learning to rank", "lambdamart", "recommendation", "recommender",
    "information retrieval", "vector search", "reranking", "re-ranking",
    # evaluation
    "ndcg", "mrr", "map@", "a/b test", "offline metric",
    # fine-tuning
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "instruction tuning",
    # core
    "python", "pytorch", "tensorflow", "transformers", "huggingface",
    "scikit-learn", "sklearn", "nlp", "bert", "gpt",
}

NICE_TO_HAVE = {
    "xgboost", "lightgbm", "spark", "kafka", "airflow", "docker", "kubernetes",
    "mlflow", "weights & biases", "wandb", "ray", "triton", "onnx",
}

# --------------------------------------------------------------------------- #
# Title classification — the single most important signal
# --------------------------------------------------------------------------- #
IDEAL_TITLE_PATTERNS = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "search engineer", "recommendation", "nlp engineer",
    "applied scientist", "applied ml", "senior data scientist",
    "staff machine learning", "staff ml", "lead ai", "lead ml",
    "senior ml", "senior ai", "senior machine learning",
    "computer vision engineer",  # borderline but keep for v1
]

ADJACENT_TITLE_PATTERNS = [
    "data scientist", "data engineer", "analytics engineer",
    "backend engineer", "software engineer", "full stack",
    "cloud engineer", "devops", "senior software",
    "senior data", "senior backend",
]

NONTECH_TITLE_PATTERNS = [
    "marketing", "sales", "hr ", "human resource", "accountant",
    "content writer", "graphic designer", "operations manager",
    "business analyst", "customer support", "civil engineer",
    "mechanical engineer", "project manager",
]

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "mindtree", "niit",
}


def classify_title(title: str) -> float:
    """Return a title-relevance score in [0, 1]."""
    t = title.lower()
    for p in NONTECH_TITLE_PATTERNS:
        if p in t:
            return 0.05
    for p in IDEAL_TITLE_PATTERNS:
        if p in t:
            return 1.0
    for p in ADJACENT_TITLE_PATTERNS:
        if p in t:
            return 0.55
    return 0.25  # unknown title — give small benefit of doubt


def title_score(cand: dict) -> float:
    prof = cand.get("profile", {})
    current = prof.get("current_title", "")
    base = classify_title(current)

    # check if their entire career is consulting — soft penalty
    history = cand.get("career_history", [])
    if history:
        companies = [r.get("company", "").lower() for r in history]
        consulting_count = sum(
            1 for c in companies
            if any(kw in c for kw in CONSULTING_COMPANIES)
        )
        if consulting_count == len(companies):
            base *= 0.6  # all consulting, no product company

    return base


# --------------------------------------------------------------------------- #
# Skill match
# --------------------------------------------------------------------------- #
def skill_score(cand: dict) -> float:
    skills = cand.get("skills", [])
    if not skills:
        return 0.0

    required_hits = 0
    nice_hits = 0
    total_weight = 0.0

    for sk in skills:
        name = sk.get("name", "").lower()
        prof = sk.get("proficiency", "beginner")
        dur  = sk.get("duration_months", 0) or 0
        end  = sk.get("endorsements", 0) or 0

        # proficiency multiplier
        prof_w = {"expert": 1.2, "advanced": 1.0,
                  "intermediate": 0.7, "beginner": 0.4}.get(prof, 0.5)

        # duration weight: saturates at 24 months
        dur_w = min(dur / 24.0, 1.0)

        # endorsement weight: saturates at 20 endorsements
        end_w = min(end / 20.0, 1.0)

        skill_weight = prof_w * (0.5 + 0.3 * dur_w + 0.2 * end_w)

        is_required = any(r in name for r in REQUIRED_SKILLS)
        is_nice     = any(r in name for r in NICE_TO_HAVE)

        if is_required:
            required_hits += skill_weight
        elif is_nice:
            nice_hits += skill_weight * 0.4
        total_weight += skill_weight

    if total_weight == 0:
        return 0.0

    # normalize: required skills are the main signal
    raw = (required_hits + nice_hits) / max(total_weight, 3.0)
    return min(raw, 1.0)


# --------------------------------------------------------------------------- #
# Experience years — smooth band centred on 5-9 years
# --------------------------------------------------------------------------- #
def experience_score(cand: dict) -> float:
    yoe = cand.get("profile", {}).get("years_of_experience", 0) or 0
    if yoe < 2:
        return 0.1
    if yoe <= 4:
        return 0.4 + (yoe - 2) * 0.1          # 0.40 → 0.60
    if yoe <= 9:
        return 1.0                              # sweet spot
    if yoe <= 12:
        return 1.0 - (yoe - 9) * 0.08          # 1.00 → 0.76
    return max(0.5, 1.0 - (yoe - 9) * 0.08)


# --------------------------------------------------------------------------- #
# Behavioral signals — availability and responsiveness
# --------------------------------------------------------------------------- #
def behavioral_score(cand: dict) -> float:
    sig = cand.get("redrob_signals", {})

    otw      = 1.0 if sig.get("open_to_work_flag") else 0.3
    resp     = float(sig.get("recruiter_response_rate") or 0.0)
    icr      = float(sig.get("interview_completion_rate") or 0.0)
    verified = (
        (1.0 if sig.get("verified_email") else 0.0) +
        (1.0 if sig.get("verified_phone") else 0.0)
    ) / 2.0
    notice   = float(sig.get("notice_period_days") or 90)
    notice_w = max(0.0, 1.0 - notice / 180.0)  # 0 days=1.0, 180 days=0.0

    b = (0.30 * resp + 0.25 * otw + 0.20 * icr
         + 0.15 * verified + 0.10 * notice_w)
    return min(b, 1.0)


# --------------------------------------------------------------------------- #
# Reasoning (simple, fact-grounded)
# --------------------------------------------------------------------------- #
def top_skills_list(cand: dict, n: int = 3) -> list:
    skills = cand.get("skills", [])
    relevant = [
        sk for sk in skills
        if any(r in sk.get("name", "").lower() for r in REQUIRED_SKILLS)
    ]
    relevant.sort(key=lambda s: s.get("endorsements", 0), reverse=True)
    names = [s["name"] for s in relevant[:n]]
    if not names:
        names = [s["name"] for s in skills[:n]]
    return names


def make_reasoning(cand: dict, rank_pos: int) -> str:
    prof  = cand.get("profile", {})
    sig   = cand.get("redrob_signals", {})
    title = prof.get("current_title", "N/A")
    yoe   = prof.get("years_of_experience", 0)
    company = ""
    hist = cand.get("career_history", [])
    for r in hist:
        if r.get("is_current"):
            company = r.get("company", "")
            break

    skills = ", ".join(top_skills_list(cand)) or "N/A"
    loc = prof.get("location", "")
    notice = sig.get("notice_period_days")

    parts = [f"{title} ({yoe:.0f} yrs)"]
    if company:
        parts[0] += f" at {company}"
    parts.append(f"relevant skills: {skills}")

    concerns = []
    if notice and notice > 60:
        concerns.append(f"{int(notice)}d notice")
    if not sig.get("open_to_work_flag"):
        concerns.append("not flagged open-to-work")
    if loc:
        parts.append(f"{loc.split(',')[0]}-based")

    text = "; ".join(parts)
    if concerns:
        text += f". Concerns: {', '.join(concerns)}"
    return text


# --------------------------------------------------------------------------- #
# Main scoring
# --------------------------------------------------------------------------- #
def score_candidate(cand: dict):
    ts = title_score(cand)
    ss = skill_score(cand)
    es = experience_score(cand)
    bs = behavioral_score(cand)

    final = W_TITLE * ts + W_SKILL * ss + W_EXP * es + W_BEHAV * bs
    return final


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #
def open_candidates(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker v1")
    ap.add_argument("--candidates", required=True,
                    help="path to candidates.jsonl(.gz)")
    ap.add_argument("--out", required=True,
                    help="output submission CSV path")
    ap.add_argument("--topn", type=int, default=TOP_N)
    args = ap.parse_args()

    heap = []  # min-heap of (score, candidate_id, cand)

    with open_candidates(args.candidates) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue

            score = score_candidate(cand)
            cid   = cand.get("candidate_id", f"UNKNOWN_{i}")

            if len(heap) < HEAP_BUFFER:
                heapq.heappush(heap, (score, cid, cand))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, cid, cand))

    # sort descending, emit top N
    top = sorted(heap, key=lambda x: (-x[0], x[1]))[:args.topn]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, (score, cid, cand) in enumerate(top, 1):
            w.writerow([
                cid,
                rank_pos,
                f"{score:.8f}",
                make_reasoning(cand, rank_pos),
            ])

    print(f"Done. Wrote {len(top)} rows to {args.out}")


if __name__ == "__main__":
    main()
