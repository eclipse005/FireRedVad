#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path


def metrics(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return {"n": n, "mae": 0.0, "max_abs": 0.0}
    err = [abs(float(a[i]) - float(b[i])) for i in range(n)]
    mae = sum(err) / n
    return {"n": n, "mae": mae, "max_abs": max(err)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True, type=Path)
    parser.add_argument("--rs", required=True, type=Path)
    args = parser.parse_args()

    py = json.loads(args.py.read_text(encoding="utf-8"))
    rs = json.loads(args.rs.read_text(encoding="utf-8"))

    print("feat_shape:", py.get("feat_shape"), rs.get("feat_shape"))
    print("feat_raw:", metrics(py.get("feat_raw", []), rs.get("feat_raw", [])))
    print("feat_cmvn:", metrics(py.get("feat_cmvn", []), rs.get("feat_cmvn", [])))
    print("probs:", metrics(py.get("probs", []), rs.get("probs", [])))

    py_ts = py.get("timestamps", [])
    rs_ts = rs.get("timestamps", [])
    print("timestamps(py):", py_ts)
    print("timestamps(rs):", rs_ts)
    if len(py_ts) == len(rs_ts) and len(py_ts) > 0:
        start_err = [abs(py_ts[i][0] - rs_ts[i][0]) for i in range(len(py_ts))]
        end_err = [abs(py_ts[i][1] - rs_ts[i][1]) for i in range(len(py_ts))]
        print(
            "timestamp_err_s:",
            {
                "start_max": max(start_err),
                "end_max": max(end_err),
                "start_mae": sum(start_err) / len(start_err),
                "end_mae": sum(end_err) / len(end_err),
            },
        )


if __name__ == "__main__":
    main()
