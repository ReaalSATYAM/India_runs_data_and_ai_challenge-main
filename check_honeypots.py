#!/usr/bin/env python3
"""
check_honeypots.py — verify the honeypot rate in your actual submission.

Reuses consistency_gate() from rank.py directly (same import, same function)
so this is checking with the exact logic that scored your submission, not a
reimplementation that could silently drift from it.

Usage:
    python check_honeypots.py --submission ./submission.csv --candidates ./candidates.jsonl

Exits non-zero if the honeypot rate is at or above the 10% disqualification
threshold, so you can wire this into a pre-submit check if you want.
"""
import argparse
import csv
import sys

import rank  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="path to submission.csv")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl(.gz)")
    args = ap.parse_args()

    with open(args.submission, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    top_ids = {r["candidate_id"]: int(r["rank"]) for r in rows}
    print(f"Checking {len(top_ids)} candidates from {args.submission}")

    found = 0
    flagged = []
    with rank.open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = __import__("json").loads(line)
            except Exception:
                continue
            cid = cand.get("candidate_id")
            if cid not in top_ids:
                continue
            found += 1
            gate, flags = rank.consistency_gate(cand)
            if gate < 1.0:
                flagged.append((top_ids[cid], cid, flags))

    print(f"Matched {found}/{len(top_ids)} submission IDs in the candidate pool")
    print()
    rate = len(flagged) / len(top_ids) if top_ids else 0.0
    print(f"Honeypot rate in top 100: {len(flagged)}/{len(top_ids)}  ({rate:.1%})")
    print(f"Disqualification threshold: 10%")
    print()

    if flagged:
        print("Flagged rows:")
        for rnk, cid, flags in sorted(flagged):
            print(f"  rank {rnk:3d}  {cid}  {flags}")
    else:
        print("No flagged candidates in your top 100. Clean.")

    print()
    if rate >= 0.10:
        print("FAIL — at or above the 10% disqualification threshold.")
        sys.exit(1)
    else:
        print("PASS — under the 10% disqualification threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()