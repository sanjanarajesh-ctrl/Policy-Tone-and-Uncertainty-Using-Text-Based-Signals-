# -*- coding: utf-8 -*-
"""
Analyze PTSI across years.
- Combines all yearly PTSI CSVs produced by run_ptsi_analysis.py
- Saves a combined CSV and a plot
- (Optional) Joins a market CSV (date, adj_close) and runs a HAR-style regression with PTSI

Usage (from project root):
  python .\scripts\analyze_ptsi.py
  python .\scripts\analyze_ptsi.py --market ".\data\processed\market_prices.csv"

Market CSV schema:
  date, adj_close
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def load_all_ptsi(processed_dir: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(processed_dir, "ptsi_*_daily.csv")))
    if not files:
        raise SystemExit(f"No yearly PTSI CSVs found in {processed_dir} (expected ptsi_*_daily.csv).")
    dfs = []
    for f in files:
        dfi = pd.read_csv(f, parse_dates=["date"])
        # basic schema guard
        required = {"date", "polarity_mean", "n_docs", "ptsi_z", "shock_flag"}
        miss = required - set(dfi.columns)
        if miss:
            raise SystemExit(f"{os.path.basename(f)} missing columns: {miss}")
        dfi["source_file"] = os.path.basename(f)
        dfs.append(dfi)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def plot_ptsi(df: pd.DataFrame, out_png: str, title: str = "PTSI z-score (all years)"):
    plt.figure(figsize=(11, 4.8))
    plt.plot(df["date"], df["ptsi_z"])
    plt.axhline(0, lw=0.8)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("PTSI (z)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


def build_market_panel(df_ptsi: pd.DataFrame, market_csv: str) -> pd.DataFrame:
    mkt = pd.read_csv(market_csv)
    # normalize names (case-insensitive)
    cols = {c.lower(): c for c in mkt.columns}
    if "date" not in cols or "adj_close" not in cols:
        raise SystemExit("Market CSV must have 'date' and 'adj_close' columns.")
    date_col = cols["date"]
    price_col = cols["adj_close"]

    mkt[date_col] = pd.to_datetime(mkt[date_col], errors="coerce")
    mkt = mkt.dropna(subset=[date_col, price_col]).set_index(date_col).sort_index()

    # log returns & a simple realized volatility proxy
    mkt["ret"] = np.log(mkt[price_col]).diff()
    # 5-day rolling stdev of returns as a simple RV proxy (annualize with sqrt(252))
    mkt["rv"] = mkt["ret"].rolling(5).std() * np.sqrt(252)

    ptsi = df_ptsi.set_index("date")[["ptsi_z", "shock_flag"]].copy()
    panel = mkt.join(ptsi, how="left")
    panel["ptsi_z"] = panel["ptsi_z"].ffill()  # carry forward last PTSI
    panel["shock_flag"] = panel["shock_flag"].fillna(0)
    return panel


def run_har_with_ptsi(panel: pd.DataFrame, out_csv: str):
    # Build HAR regressors: rv_{t-1}, weekly avg, monthly avg + ptsi_z and shock_flag
    dfm = pd.DataFrame(index=panel.index)
    dfm["rv"] = panel["rv"]
    dfm["rv_l1"] = dfm["rv"].shift(1)
    dfm["rv_w"] = dfm["rv"].rolling(5).mean().shift(1)
    dfm["rv_m"] = dfm["rv"].rolling(22).mean().shift(1)
    dfm["ptsi_z"] = panel["ptsi_z"]
    dfm["shock_flag"] = panel["shock_flag"]
    dfm = dfm.dropna()

    X = sm.add_constant(dfm[["rv_l1", "rv_w", "rv_m", "ptsi_z", "shock_flag"]])
    model = sm.OLS(dfm["rv"], X).fit()
    print(model.summary())

    # export coefficients table
    model.summary2().tables[1].to_csv(out_csv)
    print(f"Saved HAR+PTSI coefficients: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Combine yearly PTSI CSVs; plot; optional HAR regression with market CSV.")
    ap.add_argument("--processed-dir", default=os.path.join("data", "processed"),
                    help="Directory containing ptsi_*_daily.csv files (default: data/processed)")
    ap.add_argument("--out-combined", default=os.path.join("data", "processed", "PTSI_combined.csv"),
                    help="Output path for combined PTSI CSV")
    ap.add_argument("--out-plot", default=os.path.join("data", "processed", "PTSI_combined_plot.png"),
                    help="Output path for combined plot")
    ap.add_argument("--market", default=None,
                    help="Optional market CSV path with columns: date, adj_close")
    ap.add_argument("--out-har-csv", default=os.path.join("data", "processed", "har_ptsi_coeffs.csv"),
                    help="Output path for HAR regression coefficients CSV (if --market provided)")
    args = ap.parse_args()

    # 1) Load & combine yearly PTSI files
    df = load_all_ptsi(args.processed_dir)
    # Save combined CSV
    df.to_csv(args.out_combined, index=False)
    print(f"Saved combined PTSI: {args.out_combined} ({len(df)} rows)")

    # 2) Plot combined z-scores
    plot_ptsi(df, args.out_plot)

    # 3) Optional: HAR regression with a market CSV
    if args.market:
        panel = build_market_panel(df, args.market)
        if panel["rv"].notna().sum() < 30:
            print("Not enough RV observations after join; skipping HAR regression.")
        else:
            run_har_with_ptsi(panel, args.out_har_csv)


if __name__ == "__main__":
    main()
