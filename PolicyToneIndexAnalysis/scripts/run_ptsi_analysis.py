# -*- coding: utf-8 -*-
"""
Run PTSI analysis on ONE processed FR CSV at a time (VADER only, offline) with progress + length cap.
Usage (from project root):
  python .\scripts\run_ptsi_analysis.py ".\data\processed\FR-2025_fed_treasury.csv.gz" [--max-docs 50] [--max-chars 8000]
Outputs:
  data/processed/ptsi_YYYY_daily.csv
  data/processed/ptsi_YYYY_plot.png
"""

import argparse
import os
import re
import sys
import pandas as pd
import numpy as np

def infer_year_from_path(p: str) -> str:
    m = re.search(r'FR-(\d{4})', os.path.basename(p))
    return m.group(1) if m else "unknown"

def ensure_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # noqa
    except Exception:
        import nltk
        nltk.download('vader_lexicon')

def smart_truncate(text: str, max_chars: int = 8000) -> str:
    """Keep head and tail if text is long; preserves key summary and conclusions."""
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return text[:keep_head] + " ... " + text[-keep_tail:]

def main():
    ap = argparse.ArgumentParser(description="PTSI (one file at a time) with VADER sentiment (fast & robust).")
    ap.add_argument("input_csv_gz", help="Path to FR-YYYY_fed_treasury.csv.gz")
    ap.add_argument("--shock-quantile", type=float, default=0.9,
                    help="Absolute z threshold quantile for shocks (default 0.9 => |z| > 90th pct).")
    ap.add_argument("--roll-window", type=int, default=60,
                    help="Rolling window (days) for z-score baseline (default 60).")
    ap.add_argument("--min-periods", type=int, default=3,
                    help="Minimum periods for rolling stats (default 3).")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="If >0, only score the first N documents (for quick tests).")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="Cap text length per doc (keeps head+tail) to avoid slowdowns (default 8000).")
    args = ap.parse_args()

    inp = args.input_csv_gz
    if not os.path.exists(inp):
        print(f"Input not found: {inp}")
        sys.exit(1)

    year = infer_year_from_path(inp)
    out_csv = os.path.join("data", "processed", f"ptsi_{year}_daily.csv")
    out_png = os.path.join("data", "processed", f"ptsi_{year}_plot.png")

    # --- Load input
    df = pd.read_csv(inp)
    required = {"publication_date", "text"}
    if not required.issubset(df.columns):
        print(f"Input missing required columns: {required - set(df.columns)}")
        sys.exit(1)

    # --- Setup VADER
    ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # --- Clean & (optionally) trim
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.dropna(subset=["publication_date", "text"]).copy()
    if args.max_docs and args.max_docs > 0:
        df = df.iloc[:args.max_docs].copy()
    df["date"] = df["publication_date"].dt.date

    # Progress bar
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x  # fallback no-op if tqdm not installed

    # Truncate long docs to speed up VADER, keep head+tail to preserve signal
    texts = (smart_truncate(t, args.max_chars) for t in df["text"].astype(str).tolist())

    # Score with progress
    scores = []
    for t in tqdm(texts, total=len(df), desc="Scoring docs with VADER"):
        try:
            scores.append(sia.polarity_scores(t)["compound"])
        except KeyboardInterrupt:
            print("\nInterrupted by user. Partial results will be used.")
            break

    # If interrupted early, align lengths
    if len(scores) < len(df):
        df = df.iloc[:len(scores)]
    df["polarity"] = scores

    # Daily mean polarity + doc counts
    daily = (
        df.groupby("date")
          .agg(polarity_mean=("polarity", "mean"), n_docs=("polarity", "size"))
          .sort_index()
    )

    # z-score via rolling window
    roll_mean = daily["polarity_mean"].rolling(args.roll_window, min_periods=args.min_periods).mean()
    roll_std  = daily["polarity_mean"].rolling(args.roll_window, min_periods=args.min_periods).std()
    daily["ptsi_z"] = (daily["polarity_mean"] - roll_mean) / roll_std

    # Shock threshold from absolute z distribution
    q = daily["ptsi_z"].abs().quantile(args.shock-quantile if hasattr(args, "shock-quantile") else args.shock_quantile)
    daily["shock_flag"] = (daily["ptsi_z"].abs() > q).astype(int) * np.sign(daily["ptsi_z"]).fillna(0)

    # Save CSV
    daily.reset_index().rename(columns={"date": "date"}).to_csv(out_csv, index=False)
    print(f"Saved daily index: {out_csv}")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.5))
    plt.plot(pd.to_datetime(daily.index), daily["ptsi_z"])
    plt.title(f"PTSI z-score — {year} (roll={args.roll_window}, q={args.shock_quantile})")
    plt.xlabel("Date"); plt.ylabel("PTSI (z)"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_png}")

    shocks = int((daily["shock_flag"] != 0).sum())
    print(f"Days: {len(daily)} | Docs: {int(daily['n_docs'].sum())} | Shock days (|z|>{q:.3f}): {shocks}")

if __name__ == "__main__":
    main()
