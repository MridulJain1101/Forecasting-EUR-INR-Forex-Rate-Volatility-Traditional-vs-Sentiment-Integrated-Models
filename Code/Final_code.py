#!/usr/bin/env python
# coding: utf-8

# # Forecasting EUR/INR Forex Rate Volatility: Traditional model vs Sentiment Integrated model
# 
# - This thesis forecasts Volatility in EUR/INR FX rate from 2016 to 2025. 
# - Traditional GARCH model is being compared with the Sentiment Integrated GARCH model. 
# 
# **Analysis granularity is QUARTERLY, the PLOT granularity is YEARLY (2016–2025, so 10 graphs total).** 
# 
# These are two independent computations:
# - **Quarterly (the actual analysis):** for each of the 4 quarters in a year -
#   compute realized daily volatility, flag "shock days" (>2std devs above an
#   expanding historical baseline, **based on volatility**),
#   fit Traditional GARCH(1,1) and Sentiment-integrated GARCH-X, walk-forward
#   out-of-sample cross-validate, and compute RMSE / MAE / MAPE /
#   Diebold-Mariano / Directional Accuracy / QLIKE.
# - **Yearly (the plot only):** fit both models ONCE, in-sample, on the WHOLE
#   calendar year - this gives the continuous daily line for the Jan–Dec plot
#   and the reportable coefficients (omega, alpha, beta, gamma_ecb, gamma_rbi)
#   with p-values. Never mixed with the quarterly train/test analysis.
# 
# There are **three summary tables**: one row per quarter (40 rows), one row per
# year (10 rows, averaged from that year's 4 quarters), and one pooled 10-year
# table - plus the 10 yearly plots.
# 

# ## Step 1 - Importing the necessary libraries and python file with the code

# In[2]:


import sys
sys.path.insert(0, "C:/Users/Mridul Jain/Master Thesis/Main")   # folder containing fx_volatility_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import fx_volatility_pipeline as fxp

YEARS = range(2016, 2026)              # analysis years: 2016..2025 -> 10 graphs
FX_HISTORY_START_YEAR = 2015           # forex data starts here (to calculate standard deviation for 2016)
GDELT_START_YEAR = 2016                # ECB/RBI news data starts here (analysis window)


# ## Step 2 - Load ECB / RBI news data (starting 2016)

# In[3]:


ecb_full_raw = fxp.load_all_years(
    "C:/Users/Mridul Jain/Master Thesis/Main/ECB_Data/ecb_gdelt_{year}.csv",
    YEARS, label="ECB",
)
rbi_full_raw = fxp.load_all_years(
    "C:/Users/Mridul Jain/Master Thesis/Main/RBI_Data/rbi_gdelt_{year}.csv",
    YEARS, label="RBI",
)


# ## Step 3 - Fetch EUR/INR forex history (full range including 2015)
# 
# Fetched once for the whole span. Data from **2015** here so that 2016's baseline has
# a full year of prior realized-volatility observations behind it.

# In[4]:


fx_full_raw = fxp.fetch_eurinr_history("2015-01-01", "2026-01-01")
print(fx_full_raw.shape)
fx_full_raw.head()


# ## Step 4 - Building a continuous merged dataset
# 
# Parses V2Tone into `AvgTone`, aggregates to daily sentiment, computes
# `realized_var` / `realized_vol` (the squared/sqrt demeaned daily return i.e. the
# volatility proxy used throughout, replacing the old return-based approach),
# merges with price data, and builds `lag_ecb_sentiment` / `lag_rbi_sentiment`,
# all on the full continuous history, before any year-slicing happens.
# 
# **Daily sentiment is restricted to keyword-matched articles**
# (`ecb_pattern=fxp.ECB_PATTERN`, `rbi_pattern=fxp.RBI_PATTERN`) instead of
# averaging over every article in the raw GDELT file regardless of relevance,
# unfiltered averaging will diluting the sentiment score with non-financial news
# noise. `prepare_full_dataset` will print the match rate for both
# ECB and RBI to check if a lot of days end up with zero matching articles.
# 
# Note: `merged_full` will still contain 2015 rows (since forex data goes back
# that far), but their `ecb_sentiment`/`rbi_sentiment` will be 0 (no GDELT data
# loaded for 2015), since 2015 is only ever used
# for its `realized_vol` values in the baseline calculation, never
# for its sentiment.

# In[5]:


merged_full, ecb_full_raw, rbi_full_raw = fxp.prepare_full_dataset(
    ecb_full_raw, rbi_full_raw, fx_full_raw,
    ecb_pattern=fxp.ECB_PATTERN, rbi_pattern=fxp.RBI_PATTERN,
)
print(merged_full.shape)
merged_full.head()


# ## Step 5 - Loading the Loughran-McDonald financial dictionary (for the TF-IDF keyword step)

# In[6]:


financial_vocab = fxp.load_financial_vocab(
    "C:/Users/Mridul Jain/Master Thesis/Main/Loughran-McDonald_MasterDictionary_1993-2025.csv"
)


# ## Step 6 — BigQuery client (needed for the shock/normal quotation pulls)
# 
# Providing a `bq_client` for a fast dry run of the
# volatility forecasting side before spending any BigQuery quota.

# In[7]:


from google.cloud import bigquery
from google.oauth2 import service_account

key_path = "C:/Users/Mridul Jain/Master Thesis/Main/Big_Query/master-thesis-3-505019-5ccadc465d13.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)
print("Authenticated. Project:", bq_client.project)


# ## Step 7 - Sanity-check (2016) before running all 10
# 
# Runs the complete pipeline for a single year: all 4 quarters through
# `run_quarter_pipeline` (outlier flagging, GARCH fit, train/test CV, DM test,
# directional accuracy, QLIKE — the actual quarterly analysis), then the
# separate full-year in-sample fit + plot, then keyword analysis / TF-IDF —
# so you can inspect the shape of everything before committing to the full
# 10-year loop.
# 
# `history_start_date` anchors the expanding volatility baseline - it should be
# the EARLIEST date in the loaded data (start of 2015), not the start
# of the analysis window (2016), so 2016-Q1 has a real baseline behind it.

# In[9]:


history_start_date = merged_full["date"].min()
print("History starts:", history_start_date)

demo_result = fxp.run_year_pipeline(
    2016, merged_full, ecb_full_raw, rbi_full_raw, financial_vocab, history_start_date, n_std=2,
    bq_client=bq_client, pull_quotations=True, make_plot=True, verbose=True,
)

if "figure" in demo_result:
    plt.show()


# ## Step 8 - Run all 10 years (2016 through 2025 = 40 quarters analyzed, 10 plots)
# 
# Running with `pull_quotations=False` first to validate the volatility forecasting /
# outlier side across all 40 quarters quickly, then a second pass with
# quotations pulled.

# In[10]:


all_results = fxp.run_all_years(
    YEARS, merged_full, ecb_full_raw, rbi_full_raw, financial_vocab,
    history_start_date=history_start_date,
    bq_client=bq_client, pull_quotations=False, make_plot=True, verbose=True,
) 

with open("all_results.pkl", "wb") as f:
    pickle.dump(all_results, f)
print(f"\nSaved {len(all_results)} year results to all_results_without_bigq.pkl")

plt.close('all') 


#  Running with `pull_quotations=True`

# In[11]:


all_results = fxp.run_all_years(
    YEARS, merged_full, ecb_full_raw, rbi_full_raw, financial_vocab,
    history_start_date=history_start_date,
    bq_client=bq_client, pull_quotations=True, make_plot=True, verbose=True
) 

with open("all_results.pkl", "wb") as f:
    pickle.dump(all_results, f)
print(f"\nSaved {len(all_results)} year results to all_results.pkl")

plt.close('all') 


# ## Step 9 - The 10 volatility graphs (Jan 1 – Dec 31, one per year)
# 
# Each figure was already built inside `run_all_years` (Step 8) using the
# continuous full-year in-sample GARCH fit - Re-displaying them here together. Also, saving the figures in a folder to add them in the thesis.

# In[12]:


import os

# Folder for the thesis figures
fig_dir = "thesis_figures"
os.makedirs(fig_dir, exist_ok=True)

for r in all_results:
    if r.get("status") == "ok" and "figure" in r:
        fig = r["figure"]
        year = r.get("year", "unknown")

        # Display inline as before
        display(fig)

        # Save as a high-res PNG (thesis-ready) and as PDF (vector, for LaTeX)
        png_path = os.path.join(fig_dir, f"volatility_{year}.png")
        pdf_path = os.path.join(fig_dir, f"volatility_{year}.pdf")

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved {len(all_results)} charts to ./{fig_dir}/ (PNG + PDF, 300 dpi)")


# ## Step 10 - Three summary tables: quarterly detail, yearly average, and coefficients
# 
# - `build_quarterly_summary_dataframe`: one row per (year, quarter) - 40 rows,
#   the detailed analysis-level table.
# - `build_yearly_summary_dataframe`: one row per year - 10 rows, each the
#   average of that year's 4 quarters, plus the sentiment GARCH-X coefficients
#   (`gamma_ecb_sentiment_estimate`, `gamma_ecb_sentiment_pvalue`, etc.) from
#   the full-year in-sample fit.

# In[17]:


#all_results = pickle.load(f)
quarterly_df = fxp.build_quarterly_summary_dataframe(all_results)
display(quarterly_df)


# In[18]:


yearly_df = fxp.build_yearly_summary_dataframe(all_results)
display(yearly_df)


# ## Step 11 - Full coefficient tables for EVERY year and EVERY quarter
# 
# Two consolidated tables, no need to pick a `year_to_inspect` and re-run:
# - `build_yearly_coefficients_table`: every year's omega/alpha/beta(/gamma)
#    both models - stacked into one table.
# - `build_quarterly_coefficients_table`: the same, but for every one of the
#   40 quarters. Each quarter now gets its OWN in-sample GARCH fit
#   (`full_quarter_fit`, added alongside the existing walk-forward CV,
#   which only ever fit multiple *sub-windows* per quarter and never kept a
#   single reportable coefficient set at quarterly granularity).

# In[19]:


yearly_coefs = fxp.build_yearly_coefficients_table(all_results)
yearly_coefs.to_csv("yearly_coefficients.csv", index=False)
print(f"{yearly_coefs['year'].nunique()} years x 2 models -> {len(yearly_coefs)} rows "
      f"-- saved to yearly_coefficients.csv")
display(yearly_coefs)


# ### Quarterly coefficients (all 40 quarters)

# In[20]:


quarterly_coefs = fxp.build_quarterly_coefficients_table(all_results)
quarterly_coefs.to_csv("quarterly_coefficients.csv", index=False)
n_combos = quarterly_coefs[["year", "quarter"]].drop_duplicates().shape[0]
print(f"{n_combos} (year, quarter) combos x 2 models -> {len(quarterly_coefs)} rows "
      f"-- saved to quarterly_coefficients.csv")
display(quarterly_coefs)


# ## Step 12 - Top keywords per year (shock vs. normal), from volatility-based shock days

# In[23]:


import pandas as pd

def build_correct_keyword_table(all_results, top_n=3):
    rows = []
    for r in all_results:
        year = r["year"]
        tfidf = r.get("tfidf_keyword_importance")

        if tfidf is None or len(tfidf) == 0:
            rows.append({
                "year": year,
                "n_outlier_days": r.get("n_outlier_days"),
                "top_shock_keywords": "not available",
                "top_normal_keywords": "not available",
            })
            continue

        shock_top = tfidf.sort_values("coefficient", ascending=False).head(top_n)
        normal_top = tfidf.sort_values("coefficient", ascending=True).head(top_n)

        rows.append({
            "year": year,
            "n_outlier_days": r.get("n_outlier_days"),
            "top_shock_keywords": ", ".join(shock_top["keyword"]),
            "top_normal_keywords": ", ".join(normal_top["keyword"]),
        })

    return pd.DataFrame(rows)

corrected_table5 = build_correct_keyword_table(all_results)
display(corrected_table5)


# ## Step 13 - Was sentiment integration actually worth it for VOLATILITY forecasting?
# 
# Pools every QUARTER's out-of-sample points across all 10 years (40 quarters,
# ~this many out-of-fold days total) into ONE evaluation and answers:
# - Does adding lagged ECB/RBI sentiment to the GARCH **variance equation**
#   reduce RMSE/MAE/MAPE on volatility, and is the pooled Diebold-Mariano
#   difference significant?
# - Does it improve QLIKE (the standard variance-forecast loss)?
# - Is the sentiment model more responsive on shock days vs. normal days than
#   the traditional model?
# 
# Check the `quarterly_df` / `yearly_df` tables above too - a real effect
# concentrated in a handful of volatile quarters (e.g. 2020 Q1–Q2, 2022) rather
# than consistent across the decade is still worth reporting, just with that
# nuance attached.

# In[24]:


verdict = fxp.overall_verdict(all_results)


# ## Step 14 - Key metrics summary, one row per year
# 
# A compact, curated view for quickly scanning "did sentiment help this year" -
# RMSE/QLIKE/directional-accuracy for both models side by side, the pooled DM
# stat/p-value, both gamma coefficients with their p-values, and two boolean
# columns (`sentiment_helped_RMSE`, `sentiment_helped_QLIKE`) to see at
# a glance which years (if any) sentiment actually won in.

# In[25]:


key_metrics = fxp.build_key_metrics_summary(all_results)
display(key_metrics)

# Quick counts, for the thesis text:
n_years = len(key_metrics)
print(f"\nSentiment beat traditional on RMSE in {key_metrics['sentiment_helped_RMSE'].sum()}/{n_years} years")
print(f"Sentiment beat traditional on QLIKE in {key_metrics['sentiment_helped_QLIKE'].sum()}/{n_years} years")


# ## Step 15 - Shock dates table (for manually looking up what happened)
# 
# One row per shock day across all 10 years - the exact date, which quarter's
# threshold flagged it, the realized volatility and both sentiment scores that
# day, and the top ECB/RBI entity keywords already pulled for
# that date from Section 10's keyword step, as a head start before you search
# the date manually for the actual news event.

# In[26]:


shock_dates_table = fxp.build_shock_dates_table(all_results, merged_full=merged_full)
display(shock_dates_table)

# Save to CSV for easy reference while writing up the thesis:
shock_dates_table.to_csv("shock_dates_table.csv", index=False)
print(f"\n{len(shock_dates_table)} shock days across {shock_dates_table['year'].nunique()} years "
      f"-- saved to shock_dates_table.csv")


# ## Step 16 - Ljung-Box test for ARCH effects (pre-estimation diagnostic)
# 
# Before trusting the GARCH coefficients in Steps 8-11, it's worth checking
# whether volatility clustering (ARCH effects) is even present in each year's
# returns. This runs the Ljung-Box test on **squared** daily returns for each
# calendar year i.e. the standard test for autocorrelation in the squared series,
# which is what GARCH's alpha/beta are trying to capture.
# 
# - **H0:** no autocorrelation in squared returns up to lag 10 (i.e. no clustering).
# - A low p-value (< 0.05) rejects H0 → ARCH effects are present, so a GARCH fit
#   is justified for that year.
# - A high p-value → little/no clustering, and any alpha/beta estimated for
#   that year should be read with caution (there's not much genuine volatility
#   structure for the model to fit).

# In[27]:


from scipy import stats
import numpy as np

def check_arch_effects(returns, label="", lags=10):
    """
    Ljung-Box test for autocorrelation in SQUARED returns -- the standard
    pre-estimation check for whether volatility clustering (ARCH effects) is
    even present in a series. H0 = no autocorrelation up to `lags` (i.e. no
    clustering); a high p-value means we cannot reject "no clustering",
    which is exactly the condition under which GARCH's alpha/beta have
    nothing to estimate and get pushed toward their boundaries.
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    sq = (r - r.mean()) ** 2
    n = len(sq)

    acfs = []
    for k in range(1, lags + 1):
        c0 = np.sum((sq - sq.mean()) ** 2) / n
        ck = np.sum((sq[k:] - sq.mean()) * (sq[:-k] - sq.mean())) / n
        acfs.append(ck / c0)
    acfs = np.array(acfs)

    q_stat = n * (n + 2) * np.sum([acfs[k-1]**2 / (n - k) for k in range(1, lags + 1)])
    p_value = 1 - stats.chi2.cdf(q_stat, df=lags)

    print(f"{label}: Ljung-Box on squared returns (lags={lags}) -- "
          f"Q={q_stat:.3f}, p={p_value:.4f}"
          + ("  -> ARCH effects present (clustering detected)" if p_value < 0.05
             else "  -> NO significant ARCH effects (little/no clustering)"))
    return q_stat, p_value

# Run it for 2018 and a couple of contrasting years
for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    year_returns = merged_full.loc[
        (merged_full["date"] >= pd.Timestamp(year, 1, 1).date()) &
        (merged_full["date"] <= pd.Timestamp(year, 12, 31).date()),
        "daily_return_pct"
    ]
    check_arch_effects(year_returns, label=str(year))


# ## Final Summary
# 
# **Thesis:** Forecasting EUR/INR FX volatility, 2016–2025 - Traditional GARCH(1,1)
# vs. a Sentiment-integrated GARCH-X that adds ECB/RBI news sentiment as an
# exogenous regressor. Plot granularity is yearly (10 full-year in-sample fits,
# Step 9); the actual analysis is quarterly (40 quarters, walk-forward
# out-of-sample, Steps 8–11).
# 
# **ARCH-effect check (Step 16):** Ljung-Box on squared returns found
# significant clustering (p < 0.05) in 2016, 2020, 2021, 2023, 2024 and 2025,
# but not in 2017, 2018, 2019 or 2022 - so roughly half the years show the kind
# of volatility clustering GARCH is built to capture, and coefficients from the
# other years should be interpreted more cautiously.
# 
# **Pooled 10-year / 40-quarter verdict (Step 13):**
# 
# | Metric | Traditional GARCH | Sentiment-integrated GARCH-X |
# |---|---|---|
# | RMSE (vol, pp) | 0.431 | 0.458 |
# | MAE (vol, pp) | 0.334 | 0.358 |
# | QLIKE | -0.085 | 0.864 |
# | Directional accuracy | 39.4% | 46.7% |
# 
# Diebold-Mariano (pooled): stat = -2.74, p = 0.0061 - traditional GARCH is
# statistically significantly more accurate on RMSE.
# 
# - **Forecast accuracy (RMSE/MAE/QLIKE):** sentiment integration did **not**
#   help - the traditional model is both more accurate and lower-risk overall.
# - **Directional accuracy:** sentiment integration **did** help (46.7% vs
#   39.4%), i.e. it's somewhat better at calling the direction of volatility
#   moves even though it's worse at the magnitude.
# - **Year-by-year (Step 11):** sentiment beat traditional on RMSE in only
#   1 of 10 years, and on QLIKE in 3 of 10 years - the traditional model wins
#   most years, not just on average.
# - **Shock days (Step 15):** 84 shock days identified across 10 years
#   (2016 and 2025 had the most, 2017 and 2021 the fewest).
# 
# **Overall conclusion:** for point-forecast accuracy and risk-weighted loss,
# plain GARCH(1,1) outperforms the sentiment-augmented model on this dataset;
# the sentiment signal only shows a (modest) edge on directional accuracy.
# This suggests ECB/RBI news sentiment, as constructed here, carries some
# information about the *sign* of volatility changes but adds noise to the
# *magnitude*, which limits its use as a variance-forecasting input.

# In[ ]:




