#!/usr/bin/env python3
"""
audit_honeypots_full_pool.py — run consistency_gate() against EVERY candidate
in the pool (not just your top 100) and report exactly which ones fire, with
which flag, so you can manually open a sample and verify the impossibility
yourself rather than trusting the detector to grade its own homework.

Usage:
    python audit_honeypots_full_pool.py --candidates ./candidates.jsonl

Writes flagged_honeypots.csv (candidate_id, flags, title, years_of_experience)
in the current directory — open a handful of these rows directly in
sample_candidates.json / candidates.jsonl and check the math by hand.
"""
import argparse
import csv
import json
import sys
import collections

import rank  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl(.gz)")
    ap.add_argument("--out", default="flagged_honeypots.csv", help="output CSV of flagged candidates")
    args = ap.parse_args()

    n = 0
    flag_counts = collections.Counter()
    flagged_rows = []

    with rank.open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            gate, flags = rank.consistency_gate(cand)
            if gate < 1.0:
                for fl in flags:
                    flag_counts[fl] += 1
                prof = cand.get("profile", {})
                flagged_rows.append({
                    "candidate_id": cand.get("candidate_id", ""),
                    "flags": ";".join(flags),
                    "title": prof.get("current_title", ""),
                    "years_of_experience": prof.get("years_of_experience", ""),
                })
            if n % 20000 == 0:
                print(f"  ...scanned {n}", file=sys.stderr)

    print(f"Total candidates scanned: {n}")
    print(f"Total flagged (gate < 1.0): {len(flagged_rows)}")
    print()
    print("Breakdown by flag (a candidate can have more than one):")
    for fl, c in flag_counts.most_common():
        print(f"  {fl:25s} {c}")
    print()
    print(f"README's stated dataset design count: ~80 honeypots")
    print(f"rank.py's own docstring calibration claim: ~19 + ~21 + ~8 = ~48 (flags, may overlap)")
    print(f"This run found: {len(flagged_rows)}")
    print()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "flags", "title", "years_of_experience"])
        w.writeheader()
        w.writerows(flagged_rows)
    print(f"Full flagged list written to: {args.out}")
    print("Open a handful of these candidate_ids directly in candidates.jsonl")
    print("and check the flagged condition by hand before trusting this count.")


if __name__ == "__main__":
    main()