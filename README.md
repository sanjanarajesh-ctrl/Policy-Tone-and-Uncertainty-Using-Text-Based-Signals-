# Policy Fear Index (PTSI)

This repository contains the code and core processed outputs for a project that measures the **fear tone** of U.S. policy communications and studies its relationship with equity markets. The research summary and results are described in detail in [`ABOUT_THIS_WORK.md`](ABOUT_THIS_WORK.md).

Some large data files (yearly raw archives and large processed text tables) are **not committed** to this repository because they exceed GitHub’s 100 MB size limit. The code here is fully runnable against a local copy of those files.

---

## Repository Structure

```text
PolicyFearIndex/
├── ABOUT_THIS_WORK.md       # Research motivation, methods, results (paper-style summary)
├── README.md                # This file: repo usage and structure
├── data/
│   ├── market_data/
│   │   └── S&P_500_Historical_Data.csv       # Daily S&P 500 prices
│   ├── processed/
│   │   ├── PTSI_combined.csv                 # Daily Policy Fear Index (core input to regressions)
│   │   ├── PTSI_combined_plot.png
│   │   ├── fig_ptsi_vs_returns_scatter.png   # Daily scatter + fitted line
│   │   ├── fig_ptsi_rolling_beta.png         # Rolling regression β(PTSI) on returns
│   │   ├── fig_weekly_ptsi_vs_returns_scatter.png
│   │   ├── fig_weekly_ptsi_vs_vol_scatter.png
│   │   ├── fig_weekly_partial_effects_ptsi.png
│   │   ├── fig_monthly_ptsi_vs_returns_scatter.png
│   │   ├── fig_monthly_ptsi_vs_vol_scatter.png
│   │   ├── fig_monthly_partial_effects_ptsi.png
│   │   ├── ptsi_YYYY_daily.csv               # Year-specific PTSI panels (if included)
│   │   ├── ptsi_*_regression_summary.txt     # Text files with regression outputs
│   │   └── FR-YYYY_fed_treasury.csv.gz       # (Large) scored policy corpora – may be omitted from Git
│   └── raw/
│       └── ...                               # Raw Fed/Treasury text (typically zipped; not tracked if >100MB)
└── scripts/
    ├── parse_fr_bulk.py                      # Parse raw Fed/Treasury files into text
    ├── analyze_ptsi.py                       # Build daily PTSI series from text (FinBERT pipeline)
    ├── econometric_ptsi_analysis.py          # Baseline daily analysis and plots
    ├── econometric_ptsi_extended.py          # Weekly/monthly regressions and additional figures
    ├── econometric_ptsi_controls.py          # Regressions with controls and winsorization
    └── run_ptsi_analysis.py                  # Convenience script to run the full pipeline
