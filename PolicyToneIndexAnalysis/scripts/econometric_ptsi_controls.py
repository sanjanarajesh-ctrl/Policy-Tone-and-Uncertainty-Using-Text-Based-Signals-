#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
econometric_ptsi_controls.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:
    sm = None

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


def load_ptsi_and_spx() -> tuple[pd.DataFrame, pd.DataFrame]:
    ptsi = pd.read_csv(PTSI_COMBINED_PATH)
    ptsi['date'] = pd.to_datetime(ptsi['date'], errors='coerce')
    ptsi = ptsi.sort_values('date')[['date','ptsi_z']]
    spx = pd.read_csv(SP500_PATH)
    spx['Date'] = pd.to_datetime(spx['Date'], format='%m/%d/%Y', errors='coerce')
    price_col = 'Price' if 'Price' in spx.columns else ('Close' if 'Close' in spx.columns else None)
    if price_col is None:
        price_col = 'Price'
    spx[price_col] = _to_num(spx[price_col])
    spx = spx.rename(columns={'Date':'date', price_col:'spx_close'}).sort_values('date')
    spx['spx_ret'] = np.log(spx['spx_close'] / spx['spx_close'].shift(1))
    return ptsi, spx[['date','spx_close','spx_ret']]


def resample_merge(ptsi: pd.DataFrame, spx_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == 'weekly':
        rule = 'W-FRI'
    elif freq == 'monthly':
        rule = 'ME'
    else:
        raise ValueError("freq must be 'weekly' or 'monthly'")
    ptsi_r = ptsi.set_index('date').resample(rule).mean().rename(columns={'ptsi_z':'ptsi_z_'+freq})
    spx_close = spx_daily.set_index('date')['spx_close'].resample(rule).last()
    spx_ret = np.log(spx_close / spx_close.shift(1))
    out = pd.concat([ptsi_r, spx_close, spx_ret], axis=1).reset_index()
    out = out.rename(columns={'spx_close':'spx_close_'+freq})
    out['spx_ret_'+freq] = spx_ret.values
    out['ptsi_z'] = out['ptsi_z_'+freq]
    out['spx_ret'] = out['spx_ret_'+freq]
    out['spx_ret_lead1'] = out['spx_ret'].shift(-1)
    return out


def realized_volatility(spx_daily: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    df = spx_daily.copy().sort_values('date')
    df['rv_daily'] = df['spx_ret'].rolling(window).std() * np.sqrt(252)
    return df[['date','rv_daily']]


def resample_vol(rv_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    rule = 'W-FRI' if freq == 'weekly' else 'ME'
    rv = rv_daily.set_index('date')['rv_daily'].resample(rule).last().reset_index()
    rv = rv.rename(columns={'rv_daily':'rv'})
    return rv


def load_controls() -> Dict[str, pd.DataFrame]:
    ctrls = {}
    # VIX
    vix_path = DATA_MARKETS / "VIX_Historical_Data.csv"
    if vix_path.exists():
        vix = pd.read_csv(vix_path)
        date_col = 'Date' if 'Date' in vix.columns else 'DATE'
        vix[date_col] = pd.to_datetime(vix[date_col], errors='coerce', format='%m/%d/%Y') if date_col == 'Date' else pd.to_datetime(vix[date_col], errors='coerce')
        price_col = 'Close' if 'Close' in vix.columns else ('Price' if 'Price' in vix.columns else None)
        if price_col:
            vix[price_col] = _to_num(vix[price_col])
        vix = vix.rename(columns={date_col:'date', price_col:'vix_close'}).sort_values('date')
        ctrls['vix'] = vix[['date','vix_close']]
    # T10Y3M
    t103m = DATA_MARKETS / "T10Y3M_FRED.csv"
    if t103m.exists():
        sp = pd.read_csv(t103m)
        sp['DATE'] = pd.to_datetime(sp['DATE'], errors='coerce')
        sp = sp.rename(columns={'DATE':'date','T10Y3M':'t10y3m'}).sort_values('date')
        ctrls['t10y3m'] = sp[['date','t10y3m']]
    # T5YIE
    t5y = DATA_MARKETS / "T5YIE_FRED.csv"
    if t5y.exists():
        bei = pd.read_csv(t5y)
        bei['DATE'] = pd.to_datetime(bei['DATE'], errors='coerce')
        bei = bei.rename(columns={'DATE':'date','T5YIE':'t5yie'}).sort_values('date')
        ctrls['t5yie'] = bei[['date','t5yie']]
    # Optionals
    dgs10_path = DATA_MARKETS / "10Y_Yield_FRED.csv"
    if dgs10_path.exists():
        y10 = pd.read_csv(dgs10_path)
        y10['DATE'] = pd.to_datetime(y10['DATE'], errors='coerce')
        y10 = y10.rename(columns={'DATE':'date','DGS10':'y10'}).sort_values('date')
        ctrls['y10'] = y10[['date','y10']]
    tb3_path = DATA_MARKETS / "3M_TBill_FRED.csv"
    if tb3_path.exists():
        y3m = pd.read_csv(tb3_path)
        y3m['DATE'] = pd.to_datetime(y3m['DATE'], errors='coerce')
        y3m = y3m.rename(columns={'DATE':'date','TB3MS':'y3m'}).sort_values('date')
        ctrls['y3m'] = y3m[['date','y3m']]
    return ctrls


def resample_controls(ctrls: Dict[str, pd.DataFrame], freq: str) -> pd.DataFrame:
    frames = []
    rule = 'W-FRI' if freq == 'weekly' else 'ME'
    for name, df in ctrls.items():
        c = df.set_index('date').resample(rule).last().reset_index()
        frames.append(c)
    if not frames:
        return pd.DataFrame({'date': []})
    out = frames[0]
    for f in frames[1:]:
        out = pd.merge(out, f, on='date', how='outer')
    out = out.sort_values('date').reset_index(drop=True)
    return out


def winsorize(df: pd.DataFrame, cols: List[str], p: float) -> pd.DataFrame:
    if p is None or p <= 0:
        return df
    out = df.copy()
    for c in cols:
        x = out[c]
        lo, hi = x.quantile(p), x.quantile(1-p)
        out[c] = x.clip(lo, hi)
    return out


def _ols(y: pd.Series, X: pd.DataFrame, hac_lags: int = 1):
    if sm is None:
        raise ImportError("statsmodels is required")
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    return model.fit(cov_type='HAC', cov_kwds={'maxlags':hac_lags})


def tidy_params(res, model_name: str) -> pd.DataFrame:
    p = res.params.rename('coef').to_frame()
    se = res.bse.rename('se')
    pv = res.pvalues.rename('pval')
    out = p.join(se).join(pv)
    out['model'] = model_name
    out['r2'] = res.rsquared
    return out.reset_index().rename(columns={'index':'variable'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', choices=['weekly','monthly'], default='weekly')
    parser.add_argument('--winsor', type=float, default=0.0, help="Winsorize pct for returns/vol columns, e.g. 0.01")
    args = parser.parse_args()
    freq = args.freq

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    ptsi, spx_daily = load_ptsi_and_spx()
    merged = resample_merge(ptsi, spx_daily, freq=freq)

    rv_d = realized_volatility(spx_daily, window=21)
    rv = resample_vol(rv_d, freq=freq)
    merged = pd.merge(merged, rv, on='date', how='left')

    ctrls = resample_controls(load_controls(), freq=freq)
    if not ctrls.empty:
        merged = pd.merge(merged, ctrls, on='date', how='left')

    merged = winsorize(merged, cols=[c for c in ['spx_ret','rv'] if c in merged.columns], p=args.winsor)

    merged_out = DATA_PROCESSED / f"ptsi_{freq}_controls_merged.csv"
    merged.to_csv(merged_out, index=False)

    control_cols = [c for c in ['vix_close','t10y3m','t5yie','y10','y3m'] if c in merged.columns]
    X_base = ['ptsi_z'] + control_cols

    hac_lags = 1

    res1 = _ols(merged['spx_ret'], merged[X_base], hac_lags=hac_lags)
    (DATA_PROCESSED / f"ptsi_{freq}_returns_with_controls.txt").write_text(res1.summary().as_text())

    res2 = _ols(merged['spx_ret'].shift(-1), merged[X_base], hac_lags=hac_lags)
    (DATA_PROCESSED / f"ptsi_{freq}_predictive_with_controls.txt").write_text(res2.summary().as_text())

    res3 = _ols(merged['rv'], merged[X_base], hac_lags=hac_lags)
    (DATA_PROCESSED / f"ptsi_{freq}_volatility_with_controls.txt").write_text(res3.summary().as_text())

    tidy = pd.concat([
        tidy_params(res1, 'returns_t'),
        tidy_params(res2, 'returns_t_plus_1'),
        tidy_params(res3, 'volatility_t')
    ], ignore_index=True)
    tidy.to_csv(DATA_PROCESSED / f"ptsi_{freq}_controls_coefs.csv", index=False)

    # Simple partial effects plot if matplotlib available
    if plt is not None:
        try:
            control_cols_only = [c for c in control_cols if merged[c].notna().sum() > 0]
            if control_cols_only:
                Xc = sm.add_constant(merged[control_cols_only], has_constant='add')
                res_y = sm.OLS(merged['spx_ret'], Xc, missing='drop').fit()
                y_res = res_y.resid
                res_x = sm.OLS(merged['ptsi_z'], Xc, missing='drop').fit()
                x_res = res_x.resid
                plot_df = pd.DataFrame({'x_res': x_res, 'y_res': y_res}).dropna()
            else:
                plot_df = pd.DataFrame({'x_res': merged['ptsi_z'], 'y_res': merged['spx_ret']}).dropna()
            ax = plot_df.plot(kind='scatter', x='x_res', y='y_res', alpha=0.6, figsize=(7,5))
            if len(plot_df) > 2:
                coef = np.polyfit(plot_df['x_res'], plot_df['y_res'], 1)
                xs = np.linspace(plot_df['x_res'].min(), plot_df['x_res'].max(), 100)
                ys = coef[0]*xs + coef[1]
                ax.plot(xs, ys)
            ax.set_title(f"Partial effect of PTSI on {freq} returns (controls adjusted)")
            ax.set_xlabel('PTSI residual')
            ax.set_ylabel('Return residual')
            plt.tight_layout()
            plt.savefig(DATA_PROCESSED / f"fig_{freq}_partial_effects_ptsi.png", dpi=150)
            plt.close()
        except Exception:
            pass

    print(f"Done. Files written to {DATA_PROCESSED}. Controls used: {control_cols}")


if __name__ == "__main__":
    main()
