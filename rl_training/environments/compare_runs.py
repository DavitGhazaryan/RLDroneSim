#!/usr/bin/env python3
# compare_runs.py
import os, json, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root folder that contains run_* subfolders")
    ap.add_argument("--out", default="scores.csv")
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, name)
        if not (os.path.isdir(d) and name.startswith("run_")):
            continue
        mj = os.path.join(d, "metrics.json")
        if not os.path.exists(mj):
            continue
        with open(mj, "r") as f:
            m = json.load(f)
        m["run"] = name
        rows.append(m)

    if not rows:
        print("No runs found.")
        return

    df = pd.DataFrame(rows)
    # A simple composite score (lower is better): weight lateral RMSE & duration
    df["score"] = 0.6*df["lateral_rmse_m"] + 0.2*df["p95_lateral_m"] + 0.1*df["max_lateral_m"] + 0.1*(df["duration_s"]/60.0)
    df.sort_values(by=["score"], inplace=True)
    df.to_csv(args.out, index=False)
    print(df)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
