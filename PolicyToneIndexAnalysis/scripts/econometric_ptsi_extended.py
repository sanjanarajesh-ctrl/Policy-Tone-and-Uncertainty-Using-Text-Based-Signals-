#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
econometric_ptsi_extended.py

Extended econometric tests for the Policy Fear Index (PTSI):
- Weekly and monthly aggregation
- Volatility channel (realized volatility) regressions
- Optional VAR prep between PTSI and returns/volatility

Inputs (relative to project root):
  data/processed/PTSI_combined.csv
  data/market_data/S&P_500_Historical_Data.csv

Outputs (written to data/processed/):
  ptsi_market_{freq}_merged.csv
  ptsi_{freq}_returns_regression_summary.txt
  ptsi_{freq}_predictive_regression_summary.txt
  ptsi_{freq}_volatility_regression_summary.txt
  ptsi_{freq}_var_summary.txt           (if VAR runs)
  fig_{freq}_ptsi_vs_returns_scatter.png
  fig_{freq}_ptsi_vs_vol_scatter.png
  fig_{freq}_rolling_beta.png

Run from project root:
  python scripts/econometric_ptsi_extended.py --freq monthly
  python scripts/econometric_ptsi_extended.py --freq weekly
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
except Exception:
    sm = None
    VAR = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MARKETS = PROJECT_ROOT / "data" / "market_data"

PTSI_COMBINED_PATH = DATA_PROCESSED / "PTSI_combined.csv"
SP500_PATH = DATA_MARKETS / "S&P_500_Historical_Data.csv"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ptsi = pd.read_csv(PTSI_COMBINED_PATH)
    ptsi['date'] = pd.to_datetime(ptsi['date'], errors='coerce')
    ptsi = ptsi.sort_values('date')
    keep = [c for c in ['date','ptsi_z','polarity_mean','n_docs'] if c in ptsi.columns]
    ptsi = ptsi[keep]

    spx = pd.read_csv(SP500_PATH)
    spx['Date'] = pd.to_datetime(spx['Date'], format='%m/%d/%Y', errors='coerce')
    for col in ['Price','Open','High','Low']:
        if col in spx.columns:
            spx[col] = _to_num(spx[col])
    spx = spx.rename(columns={'Date':'date','Price':'spx_close'}).sort_values('date')
    spx['spx_ret'] = np.log(spx['spx_close'] / spx['spx_close'].shift(1))
    return ptsi, spx[['date','spx_close','spx_ret']]


def aggregate(ptsi: pd.DataFrame, spx: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == 'weekly':
        rule = 'W-FRI'
    elif freq == 'monthly':
        rule = 'ME'
    else:
        raise ValueError("freq must be 'weekly' or 'monthly'")

    ptsi_r = (
        ptsi.set_index('date')
            .resample(rule)
            .agg({'ptsi_z':'mean', 'polarity_mean':'mean', 'n_docs':'sum'})
            .rename(columns={'ptsi_z':'ptsi_z_'+freq})
    )
    spx_close = spx.set_index('date')['spx_close'].resample(rule).last()
    spx_ret = np.log(spx_close / spx_close.shift(1))
    out = pd.concat([ptsi_r, spx_close, spx_ret], axis=1).reset_index()
    out = out.rename(columns={'index':'date','spx_close':'spx_close_'+freq})
    out['spx_ret_'+freq] = spx_ret.values
    out['ptsi_z'] = out['ptsi_z_'+freq]
    out['spx_ret'] = out['spx_ret_'+freq]
    out['d_ptsi_z'] = out['ptsi_z'].diff()
    out['spx_ret_lead1'] = out['spx_ret'].shift(-1)
    return out


def realized_volatility(spx_daily: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    df = spx_daily.copy().sort_values('date')
    df['rv'] = df['spx_ret'].rolling(window).std() * np.sqrt(252)  # annualized
    return df[['date','rv']]


def merge_with_vol(agg: pd.DataFrame, spx_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    rv = realized_volatility(spx_daily, window=21)
    if freq == 'weekly':
        rv = rv.set_index('date').resample('W-FRI').last().reset_index()
    else:
        rv = rv.set_index('date').resample('ME').last().reset_index()
    merged = pd.merge(agg, rv, on='date', how='left')
    merged = merged.rename(columns={'rv':'rv_'+freq})
    merged['rv'] = merged['rv_'+freq]
    return merged


def _ols(y: pd.Series, X: pd.DataFrame):
    if sm is None:
        raise ImportError("statsmodels is required for regression")
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags':1})
    return res


def write_summary(res, path: Path):
    path.write_text(res.summary().as_text())


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path):
    if plt is None:
        return
    clean = df[[x_col, y_col]].dropna()
    ax = clean.plot(kind='scatter', x=x_col, y=y_col, alpha=0.6, figsize=(8,6))
    if len(clean) > 2:
        coef = np.polyfit(clean[x_col], clean[y_col], 1)
        x = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
        y = coef[0]*x + coef[1]
        ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def rolling_beta(df: pd.DataFrame, window: int, out_png: Path):
    if plt is None or sm is None:
        return
    betas, dates = [], []
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i][['spx_ret','ptsi_z']].dropna()
        if len(sub) < max(3, int(window*0.8)):
            betas.append(np.nan)
            dates.append(df.iloc[i]['date'])
            continue
        try:
            res = _ols(sub['spx_ret'], sub[['ptsi_z']])
            betas.append(res.params.get('ptsi_z', np.nan))
        except Exception:
            betas.append(np.nan)
        dates.append(df.iloc[i]['date'])
    rb = pd.DataFrame({'date': dates, 'beta': betas}).dropna()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(rb['date'], rb['beta'])
    ax.set_title(f'Rolling beta ({window} windows): spx_ret ~ ptsi_z')
    ax.set_xlabel('Date')
    ax.set_ylabel('beta')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def maybe_var(df: pd.DataFrame, out_path: Path):
    if VAR is None:
        return
    clean = df[['spx_ret','ptsi_z']].dropna()
    if len(clean) < 40:
        return
    try:
        model = VAR(clean)
        res = model.fit(maxlags=1, ic='aic')
        out_path.write_text(str(res.summary()))
    except Exception as e:
        out_path.write_text(f"VAR failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', choices=['weekly','monthly'], default='monthly')
    args = parser.parse_args()
    freq = args.freq

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    ptsi, spx_daily = load_data()
    agg = aggregate(ptsi, spx_daily, freq=freq)
    agg = merge_with_vol(agg, spx_daily, freq=freq)
    (DATA_PROCESSED / f"ptsi_market_{freq}_merged.csv").write_text(agg.to_csv(index=False))

    # Regressions
    res_ret = _ols(agg['spx_ret'], agg[['ptsi_z']])
    write_summary(res_ret, DATA_PROCESSED / f"ptsi_{freq}_returns_regression_summary.txt")

    res_pred = _ols(agg['spx_ret'].shift(-1), agg[['ptsi_z']])
    write_summary(res_pred, DATA_PROCESSED / f"ptsi_{freq}_predictive_regression_summary.txt")

    res_vol = _ols(agg['rv'], agg[['ptsi_z']])
    write_summary(res_vol, DATA_PROCESSED / f"ptsi_{freq}_volatility_regression_summary.txt")

    # Plots
    plot_scatter(agg, 'ptsi_z', 'spx_ret', f"S&P {freq} log returns vs PTSI (z)", DATA_PROCESSED / f"fig_{freq}_ptsi_vs_returns_scatter.png")
    plot_scatter(agg, 'ptsi_z', 'rv', f"Realized volatility ({freq}) vs PTSI (z)", DATA_PROCESSED / f"fig_{freq}_ptsi_vs_vol_scatter.png")
    try:
        rolling_beta(agg, window=6, out_png=DATA_PROCESSED / f"fig_{freq}_rolling_beta.png")
    except Exception:
        pass

    maybe_var(agg, DATA_PROCESSED / f"ptsi_{freq}_var_summary.txt")

    print(f"Done. Outputs written to {DATA_PROCESSED} for freq={freq}")


if __name__ == "__main__":
    main()
