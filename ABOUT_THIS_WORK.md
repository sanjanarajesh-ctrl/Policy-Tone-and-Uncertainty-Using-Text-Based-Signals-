# Measuring Policy Tone and Uncertainty Using Text-Based Signals in the United States

This document describes the research question, data, empirical strategy, and main findings for the **Policy Fear Index (PTSI)** project. The code for this work lives in the current repository; some large data files are not tracked in Git for size reasons and are noted in the main `README.md`.

---

## 1. Research Question and Motivation

The project asks a simple question:

> *Does the language used by U.S. policymakers contain information about future equity market performance?*

Central banks and Treasuries do more than change interest rates and balance sheets. They also communicate, and that communication shapes expectations. When official language turns more cautious or fearful, investors may infer that policymakers perceive higher downside risks, even before any concrete policy move occurs.

Existing measures of policy-related uncertainty, such as news-based indices, focus on how *media* cover policy. This project instead measures the *tone of policy documents themselves*. The aim is to build a daily index of "policy fear" based on Federal Reserve and U.S. Treasury communications and to test whether this index has predictive content for U.S. equity returns and realized volatility.

---

## 2. Data Overview

### 2.1 Policy Text Corpus

The Policy Fear Index is constructed from the full text of:

- **Federal Reserve communications**
  - FOMC post-meeting statements  
  - FOMC minutes  
  - Board of Governors policy-relevant press releases  
  - Speeches and testimonies by Board members and Reserve Bank presidents that discuss macroeconomic conditions, financial stability, or the stance of monetary policy

- **U.S. Treasury communications**
  - Press releases and statements related to fiscal, financial, or regulatory policy  
  - Speeches by the Treasury Secretary and senior officials that comment on the economy or markets

Documents are taken from official online archives and archived locally by institution and year (for example `data/raw/fed/2019/…`, `data/raw/treasury/2021/…`). Non-substantive items (ceremonial remarks, logistical notices) are excluded at the scraping stage.

For GitHub size limits, the largest raw and processed text archives (e.g. `FR-YYYY_fed_treasury.csv.gz` and yearly `.zip` bundles) are **not committed**. The scripts in this repository reproduce `PTSI_combined.csv` and the figures from those local files.

### 2.2 Market Data

Market data consist of daily S&P 500 index levels stored in:

- `data/market_data/S&P_500_Historical_Data.csv`

From this file, the code constructs:

- Daily log returns  
- A 21-day rolling realized volatility measure (annualized)

These are merged with the daily Policy Fear Index and resampled to weekly (Friday) and monthly (month-end) frequencies for the econometric analysis.

---

## 3. Construction of the Policy Fear Index (PTSI)

### 3.1 Text Pre-processing

The script `scripts/analyze_ptsi.py` performs the following steps:

1. **Cleaning and normalization**  
   HTML remnants and boilerplate are removed, text is lowercased, and sentences are segmented.

2. **Procedural filter**  
   Sentences consisting purely of procedural language (e.g. attendance, vote tallies, scheduling) are dropped to prevent them from diluting the tone signal.

3. **Sentence-level sentiment (FinBERT)**  
   Each remaining sentence is passed through the FinBERT model, which returns probabilities of negative, neutral, and positive sentiment. A continuous polarity score is defined as  
   `polarity = p_pos – p_neg`.

4. **Document-level tone**  
   For each document, sentence polarities are averaged into a document-level score. Strongly negative sentences receive modest extra weight so that clearly fearful language has more influence than mild reassurance.

5. **Daily aggregation and standardization**  
   For each calendar date, document scores are averaged across all Federal Reserve and Treasury communications released that day. This yields a daily polarity series and a document count. The polarity series is then standardized (z-score) over the full sample to form the **Policy Fear Index**, stored as `ptsi_z` in `data/processed/PTSI_combined.csv`.

No smoothing or winsorization is applied to `ptsi_z` in the baseline. Smoothing is used only for descriptive plots.

---

## 4. Empirical Strategy

### 4.1 Baseline Specifications

The econometric work relates policy tone to returns and realized volatility using simple linear models of the form

\[
Y_t = \alpha + \beta \, PTSI_t + \gamma' X_t + \epsilon_t,
\]

where \(Y_t\) is:

- \(r_t\): same-period log return,  
- \(r_{t+1}\): next-period log return (predictive specification), or  
- \(RV_t\): realized volatility.

\(PTSI_t\) is the standardized Policy Fear Index, and \(X_t\) is a vector of macro-financial controls:

- VIX (implied equity volatility)  
- 10-year minus 3-month Treasury term spread  
- 5-year TIPS breakeven inflation

Models are estimated on daily, weekly, and monthly panels. The weekly and monthly regressions are the main focus, as they better match realistic investor reaction horizons.

All regressions use ordinary least squares with Newey–West (HAC) standard errors (lag 1 for weekly and monthly data). A simple VAR(1) system in \((r_t, PTSI_t)\) is also estimated as a dynamic check on directionality.

### 4.2 Identification Logic

The identification argument rests on timing and institutional behavior.

- Policy documents reflect deliberations and staff assessments formed before release. Same-day trading cannot change the wording of a speech that is already drafted.
- Policymakers respond to broader macroeconomic developments, not to individual daily return realizations; tone is driven by perceived economic risk rather than by intraday volatility.
- The main predictive specification relates \(PTSI_t\) to \(r_{t+1}\), so that tone precedes the returns being explained.
- In VAR estimates, lagged PTSI helps forecast returns, whereas lagged returns do not materially improve forecasts of PTSI, which is consistent with tone acting as a communication shock rather than a passive reflection of market moves.

These features limit reverse causality and support interpreting \(\beta\) in the predictive regressions as the effect of policy tone on subsequent market behavior.

### 4.3 Robustness

Two robustness dimensions are implemented:

1. **Controls.**  
   Regressions are run with and without macro-financial controls. The PTSI coefficient is stable across these specifications, indicating that policy tone carries information beyond VIX, term spreads, and breakeven inflation.

2. **Winsorization.**  
   In additional runs, returns and realized volatility are winsorized at the 1st and 99th percentiles. The PTSI coefficient in predictive weekly regressions is nearly unchanged, suggesting that the results are not driven solely by extreme crisis episodes.

---

## 5. Main Findings (Condensed)

Detailed regression outputs are saved in `data/processed/*.txt`. The main qualitative patterns are:

1. **No strong daily effect.**  
   On the daily panel, the PTSI has little effect on same-day returns or realized volatility. This is consistent with the idea that policy tone takes time to be interpreted and to influence positions.

2. **Short-run predictive power for returns.**  
   On the weekly panel, more fearful policy tone today is associated with lower equity returns next week. A one-standard-deviation rise in PTSI is followed by a drop in the S&P 500 of roughly 0.5–0.6 percentage points. This effect is robust to controls and to winsorization.

3. **Medium-run mean reversion.**  
   At the monthly horizon, fearful tone coincides with weaker same-month returns but a modest positive coefficient in the next-month regression, consistent with partial rebound as uncertainty is resolved.

4. **Limited volatility channel.**  
   Across frequencies, PTSI has little systematic effect on realized volatility once controls are included. Policy tone appears to shift expected direction of returns more than the dispersion of returns.

In summary, the Policy Fear Index captures a communication channel in which fearful language from the Federal Reserve and Treasury precedes short-run equity weakness that is gradually reversed as new information arrives.

---

## 6. Possible Extensions

The current implementation focuses on aggregate S&P 500 returns and a unified index based on all Fed and Treasury communications. Natural extensions include:

- Intraday event-study windows around major releases  
- Separate tone indices for FOMC statements, minutes, and Treasury communications  
- Topic-specific tone measures (e.g. inflation, financial stability, labor market) and their differential effects on sectoral returns

These extensions are straightforward to implement within the existing data and code structure and are left for future work.
