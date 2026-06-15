#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def find_first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None


def window(vals, center, radius):
    s = max(0, center - radius)
    e = min(len(vals), center + radius + 1)
    return s, e, vals[s:e]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path)
    parser.add_argument("--rs", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=20)
    args = parser.parse_args()

    py = json.loads(args.py.read_text(encoding="utf-8"))
    rs = json.loads(args.rs.read_text(encoding="utf-8"))

    stages = [
        "probs",
        "smoothed_probs",
        "binary_preds",
        "state_decisions",
        "fixed_decisions",
        "merged_decisions",
        "extended_decisions",
        "final_decisions",
    ]

    for st in stages:
        if st not in py or st not in rs:
            continue
        a, b = py[st], rs[st]
        n = min(len(a), len(b))
        if st.endswith("decisions") or st == "binary_preds":
            idx = find_first_diff(a, b)
            print(f"{st}: first_diff={idx}, len_py={len(a)}, len_rs={len(b)}")
            if idx is not None:
                s, e, wa = window(a, idx, args.radius)
                _, _, wb = window(b, idx, args.radius)
                print("  frame_range", [s, e])
                print("  py ", wa)
                print("  rs ", wb)
        else:
            max_abs = 0.0
            max_idx = 0
            for i in range(n):
                d = abs(float(a[i]) - float(b[i]))
                if d > max_abs:
                    max_abs = d
                    max_idx = i
            print(f"{st}: max_abs={max_abs:.8f} at idx={max_idx}, len_py={len(a)}, len_rs={len(b)}")


if __name__ == "__main__":
    main()
