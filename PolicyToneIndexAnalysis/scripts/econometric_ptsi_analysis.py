#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
econometric_ptsi_analysis.py

Quantifies the relationship between the Policy Fear Index (PTSI) and S&P 500 returns.

Inputs (relative to project root):
    - data/processed/PTSI_combined.csv
        Expected columns: ['date','polarity_mean','n_docs','ptsi_z','shock_flag','source_file']
    - data/market_data/S&P_500_Historical_Data.csv
        Expected columns: ['Date','Price','Open','High','Low','Vol.','Change %']
        Notes: numbers use commas as thousands separators, dates are mm/dd/YYYY,
               order is typically descending (latest first).

Outputs:
    - data/processed/ptsi_market_merged.csv
    - data/processed/ptsi_returns_regression_summary.txt
    - data/processed/ptsi_predictive_regression_summary.txt
    - data/processed/ptsi_differenced_regression_summary.txt
    - data/processed/ptsi_rolling_beta.csv
    - data/processed/fig_ptsi_vs_returns_scatter.png
    - data/processed/fig_ptsi_rolling_beta.png

Run:
    python scripts/econometric_ptsi_analysis.py
"""
from __future__ import annotations

import os
import sys
import math
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Optional imports (only needed for regressions and plots)
try:
    import statsmodels.api as sm
    import statsmodels.tools.tools as sm_tools
except Exception as e:
    sm = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# --------------------------
# Utility helpers
# --------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MARKETS = PROJECT_ROOT / "data" / "market_data"

PTSI_COMBINED_PATH = DATA_PROCESSED / "PTSI_combined.csv"
SP500_PATH = DATA_MARKETS / "S&P_500_Historical_Data.csv"


def _ensure_paths() -> None:
    if not PTSI_COMBINED_PATH.exists():
        raise FileNotFoundError(f"Missing: {PTSI_COMBINED_PATH}")
    if not SP500_PATH.exists():
        raise FileNotFoundError(f"Missing: {SP500_PATH}")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _read_ptsi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expected schema from your combined file
    # date: YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Keep the key columns
    cols = ['date','ptsi_z','polarity_mean','n_docs','shock_flag','source_file']
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.sort_values('date')
    return df


def _to_num(x: pd.Series) -> pd.Series:
    # Convert strings like '5,123.45' to float
    return pd.to_numeric(x.astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')


def _read_sp500(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    # investor.com schema: Date (mm/dd/YYYY), columns as strings with commas; Vol can be NaN
    m['Date'] = pd.to_datetime(m['Date'], format='%m/%d/%Y', errors='coerce')
    for col in ['Price','Open','High','Low']:
        if col in m.columns:
            m[col] = _to_num(m[col])
    # Change % like '0.45%'
    if 'Change %' in m.columns:
        m['Change_pct'] = pd.to_numeric(m['Change %'].str.replace('%','', regex=False), errors='coerce') / 100.0
    # Sort ascending by date
    m = m.sort_values('Date').rename(columns={'Date':'date','Price':'spx_close'})
    # Compute log returns
    m['spx_ret'] = np.log(m['spx_close'] / m['spx_close'].shift(1))
    return m[['date','spx_close','spx_ret'] + ([ 'Change_pct' ] if 'Change_pct' in m.columns else [])]


def _merge(ptsi: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    # Inner join on daily dates
    merged = pd.merge(ptsi, spx, on='date', how='inner')
    # Construct transformed variables
    merged['d_ptsi_z'] = merged['ptsi_z'].diff()
    # Lead 1-day return for predictive regression (return_{t+1})
    merged['spx_ret_lead1'] = merged['spx_ret'].shift(-1)
    return merged


def _ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True):
    if sm is None:
        raise ImportError("statsmodels is required for regression. Please 'pip install statsmodels'.")
    if add_const:
        X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    return model.fit(cov_type='HAC', cov_kwds={'maxlags':5})  # Newey-West (HAC) with small lag as default


def _write_summary(res, out_path: Path) -> None:
    out_path.write_text(res.summary().as_text())


def _rolling_beta(df: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Rolling OLS of spx_ret ~ ptsi_z with a given window (~6 trading months by default)."""
    out = []
    if sm is None:
        return pd.DataFrame()
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i]
        try:
            res = _ols(sub['spx_ret'], sub[['ptsi_z']])
            out.append({'date': df.iloc[i]['date'], 'beta': res.params.get('ptsi_z', np.nan)})
        except Exception:
            out.append({'date': df.iloc[i]['date'], 'beta': np.nan})
    return pd.DataFrame(out)


def _plot_scatter(df: pd.DataFrame, out_png: Path) -> None:
    if plt is None:
        return
    ax = df.plot(kind='scatter', x='ptsi_z', y='spx_ret', alpha=0.4, figsize=(8,6))
    # Fit line for visualization only
    clean = df[['ptsi_z','spx_ret']].dropna()
    if len(clean) > 2:
        coef = np.polyfit(clean['ptsi_z'], clean['spx_ret'], 1)
        x = np.linspace(clean['ptsi_z'].min(), clean['ptsi_z'].max(), 100)
        y = coef[0]*x + coef[1]
        ax.plot(x, y)
    ax.set_title('S&P 500 Daily Log Returns vs. PTSI (z)')
    ax.set_xlabel('PTSI (z-score)')
    ax.set_ylabel('S&P 500 log return')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_rolling_beta(rb: pd.DataFrame, out_png: Path) -> None:
    if plt is None or rb.empty:
        return
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(rb['date'], rb['beta'])
    ax.set_title('Rolling β: spx_ret ~ ptsi_z')
    ax.set_xlabel('Date')
    ax.set_ylabel('β (PTSI)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    _ensure_paths()

    ptsi = _read_ptsi(PTSI_COMBINED_PATH)
    spx = _read_sp500(SP500_PATH)

    merged = _merge(ptsi, spx)
    merged_out = DATA_PROCESSED / "ptsi_market_merged.csv"
    merged.to_csv(merged_out, index=False)

    # -------- Baseline contemporaneous: ret_t ~ ptsi_z_t
    try:
        res_contemp = _ols(merged['spx_ret'], merged[['ptsi_z']])
        _write_summary(res_contemp, DATA_PROCESSED / "ptsi_returns_regression_summary.txt")
    except Exception as e:
        print(f"[WARN] contemporaneous regression failed: {e}")

    # -------- Predictive: ret_{t+1} ~ ptsi_z_t
    try:
        res_pred = _ols(merged['spx_ret_lead1'], merged[['ptsi_z']])
        _write_summary(res_pred, DATA_PROCESSED / "ptsi_predictive_regression_summary.txt")
    except Exception as e:
        print(f"[WARN] predictive regression failed: {e}")

    # -------- Differenced: ret_t ~ Δptsi_z_t
    try:
        res_diff = _ols(merged['spx_ret'], merged[['d_ptsi_z']])
        _write_summary(res_diff, DATA_PROCESSED / "ptsi_differenced_regression_summary.txt")
    except Exception as e:
        print(f"[WARN] differenced regression failed: {e}")

    # -------- Rolling β
    try:
        rb = _rolling_beta(merged, window=126)  # ~ half year of trading days
        if not rb.empty:
            rb_out = DATA_PROCESSED / "ptsi_rolling_beta.csv"
            rb.to_csv(rb_out, index=False)
    except Exception as e:
        print(f"[WARN] rolling beta failed: {e}")
        rb = pd.DataFrame()

    # -------- Plots
    try:
        _plot_scatter(merged, DATA_PROCESSED / "fig_ptsi_vs_returns_scatter.png")
    except Exception as e:
        print(f"[WARN] scatter plot failed: {e}")

    try:
        if not rb.empty:
            _plot_rolling_beta(rb, DATA_PROCESSED / "fig_ptsi_rolling_beta.png")
    except Exception as e:
        print(f"[WARN] rolling beta plot failed: {e}")

    print("Done. Outputs written to:", DATA_PROCESSED)


if __name__ == "__main__":
    main()
