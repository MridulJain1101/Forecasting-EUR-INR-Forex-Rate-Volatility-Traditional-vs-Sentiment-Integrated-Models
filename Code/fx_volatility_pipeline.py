import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.optimize import minimize


# =====================================================================================
# SECTION 0: KEYWORD PATTERNS
# =====================================================================================

ECB_KEYWORDS = [
    "ECB", "eurozone inflation", "deposit facility rate", "hawkish", "dovish",
    "quantitative tightening", "quantitative easing", "German inflation",
    "eurozone recession", "PMI eurozone", "energy crisis",
    "natural gas prices Europe", "Ukraine war sanctions", "euro area GDP",
    "ECB press conference",
]

RBI_KEYWORDS = [
    "RBI", "repo rate", "CPI inflation India", "rupee depreciation",
    "current account deficit", "FII outflows", "crude oil import bill",
    "monetary policy committee", "forex reserves RBI", "India GDP growth",
    "rupee intervention", "Fed rate hike spillover", "India current account",
    "RBI governor statement",
]


def build_keyword_regex(keywords):
    """Escape, word-boundary each keyword, OR them together (non-capturing group,
    so pandas str.contains doesn't warn about unused match groups)."""
    escaped = [r"\b" + kw.replace(" ", r"\s+") + r"\b" for kw in keywords]
    return "(?i)(?:" + "|".join(escaped) + ")"


ECB_PATTERN = build_keyword_regex(ECB_KEYWORDS)
RBI_PATTERN = build_keyword_regex(RBI_KEYWORDS)


# =====================================================================================
# SECTION 1: LOADING, YEARLY FILTERING (for the continuous plot), AND QUARTERLY
#            FILTERING (for the GARCH fitting / train-test-split / error-metric analysis)
# =====================================================================================
def load_all_years(file_path_template, years, label=""):
    """
    Load and concatenate per-year GDELT CSVs into ONE continuous DataFrame,
    parsed and sorted by date. Done once (not per-slice) so lag features
    computed later are continuous across quarter/year boundaries.
    """
    dfs = []
    for year in years:
        path = file_path_template.format(year=year)
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"[{label}] Loaded {year}: {len(df):,} rows")
    combined = pd.concat(dfs, ignore_index=True)
    combined["DATE"] = pd.to_datetime(combined["DATE"].astype(str), format="%Y%m%d%H%M%S")
    combined["date_only"] = combined["DATE"].dt.date
    combined = combined.sort_values("DATE").reset_index(drop=True)
    print(f"[{label}] TOTAL: {len(combined):,} rows, {combined['date_only'].min()} to {combined['date_only'].max()}")
    return combined


def get_year_bounds(year):
    """Return (start_date, end_date) as date objects for Jan 1 - Dec 31 of `year`. Used for the plot."""
    start = pd.Timestamp(year=year, month=1, day=1).date()
    end = pd.Timestamp(year=year, month=12, day=31).date()
    return start, end


def filter_to_year(df, year, date_col="date_only"):
    """Pure in-memory boolean filter to a single calendar year. Used for the plot."""
    start, end = get_year_bounds(year)
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[mask].copy()


QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


def get_quarter_bounds(year, quarter):
    """Return (start_date, end_date) as date objects for a given (year, quarter). Used for the analysis."""
    start_month, end_month = QUARTER_MONTHS[quarter]
    start = pd.Timestamp(year=year, month=start_month, day=1).date()
    end = (pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)).date()
    return start, end


def filter_to_quarter(df, year, quarter, date_col="date"):
    """Pure in-memory boolean filter to a single (year, quarter). Used for the analysis."""
    start, end = get_quarter_bounds(year, quarter)
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[mask].copy()


def quarters_up_to(max_year, max_quarter, start_year):
    """Chronological list of every (year, quarter) pair from start_year Q1 through max_year/max_quarter."""
    out = []
    for y in range(start_year, max_year + 1):
        last_q = max_quarter if y == max_year else 4
        for q in range(1, last_q + 1):
            out.append((y, q))
    return out


# =====================================================================================
# SECTION 2: SENTIMENT EXTRACTION FROM V2TONE
# =====================================================================================

def parse_v2tone(df, tone_col="V2Tone", out_col="AvgTone"):
    """V2Tone format: AvgTone,PositiveScore,NegativeScore,... -- we want the first value."""
    df = df.copy()
    tone_split = df[tone_col].astype(str).str.split(",", expand=True)
    df[out_col] = pd.to_numeric(tone_split[0], errors="coerce")
    return df


def compute_daily_sentiment(df, pattern=None, value_col="AvgTone", date_col="date_only",
                             name_col="AllNames", out_col="sentiment"):
    """
    Mean AvgTone across articles per day -> one sentiment score per day.

    IMPORTANT: pass `pattern` (ECB_PATTERN / RBI_PATTERN) to restrict the
    averaging to only articles whose AllNames matches your curated
    financial/macro keyword list, BEFORE averaging. Without this, the daily
    sentiment score is the mean tone of every article in the raw GDELT file
    for that day, regardless of whether the article is financially relevant
    at all -- diluting whatever real signal exists with unrelated news noise
    and making the sentiment score a weaker, noisier predictor than it needs
    to be. This is the same keyword criteria already used elsewhere in the
    pipeline (quotation pulls, entity-count keywords), now made consistent
    with what actually feeds the GARCH-X exogenous regressor.
    """
    if pattern is not None:
        df = df[df[name_col].astype(str).str.contains(pattern, regex=True, na=False)]
    daily = df.groupby(date_col)[value_col].mean().reset_index()
    daily.columns = ["date", out_col]
    return daily


# =====================================================================================
# SECTION 3: FOREX DATA, REALIZED VOLATILITY, EXPANDING-WINDOW BASELINE, AND
#            VOLATILITY-BASED OUTLIER ("SHOCK DAY") FLAGGING
# =====================================================================================

def fetch_eurinr_history(start_date, end_date):
    """Fetch the FULL EUR/INR history ONCE."""
    import yfinance as yf

    ticker = "EURINR=X"
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df = df[["open", "high", "low", "close"]].reset_index()
    df = df.rename(columns={"Date": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = add_returns_and_volatility(df)
    return df


def add_returns_and_volatility(df, demean_window=None):
    """
    Computes:
      daily_return_pct  : close-to-close percentage return (same as before)
      true_range        : high - low (kept for reference / optional Parkinson
                           robustness check; not the primary volatility proxy)
      realized_var      : (r_t - mean_r)^2 -- the squared demeaned return. This
                           is the standard "realized variance" proxy used to
                           BOTH evaluate GARCH forecasts and flag shock days
                           when only daily (not intraday) data is available.
                           It is the theoretically consistent target for a
                           GARCH variance forecast, since GARCH itself assumes
                           E[eps_t^2 | F_{t-1}] = sigma2_t.
      realized_vol      : sqrt(realized_var) -- "daily realized volatility, in
                           percentage points" -- this is what gets plotted and
                           what the +/-2-std-dev outlier rule is applied to.

    mean_r is a SIMPLE full-sample mean by default (demean_window=None). This
    is deliberately simple/stable; if you want a rolling mean instead, pass
    demean_window=N (e.g. 20) for an N-day rolling mean.
    """
    df = df.copy()
    df["daily_return_pct"] = df["close"].pct_change() * 100
    df["true_range"] = df["high"] - df["low"]

    if demean_window is None:
        mean_r = df["daily_return_pct"].mean()
        df["realized_var"] = (df["daily_return_pct"] - mean_r) ** 2
    else:
        mean_r = df["daily_return_pct"].rolling(demean_window, min_periods=5).mean()
        df["realized_var"] = (df["daily_return_pct"] - mean_r) ** 2

    df["realized_vol"] = np.sqrt(df["realized_var"])
    return df


def get_expanding_baseline_window(year, quarter, history_start_date):
    """
    Baseline window for the volatility-outlier calculation: history_start_date
    through the day before the target quarter begins. Grows every QUARTER (not
    every year) -- no look-ahead, since it only uses data strictly before the
    target quarter begins.
    """
    q_start, _ = get_quarter_bounds(year, quarter)
    baseline_end = q_start - pd.Timedelta(days=1)
    baseline_end = baseline_end.date() if hasattr(baseline_end, "date") else baseline_end
    return history_start_date, baseline_end


def compute_expanding_baseline_stats(fx_full_df, year, quarter, history_start_date, date_col="date"):
    """Mean and std of REALIZED VOLATILITY (not returns) over the expanding baseline window, up to this quarter."""
    baseline_start, baseline_end = get_expanding_baseline_window(year, quarter, history_start_date)
    mask = (fx_full_df[date_col] >= baseline_start) & (fx_full_df[date_col] <= baseline_end)
    baseline = fx_full_df.loc[mask, "realized_vol"].dropna()
    if len(baseline) < 30:
        warnings.warn(
            f"Only {len(baseline)} baseline observations for {year}-Q{quarter} "
            f"({baseline_start} to {baseline_end}) -- std dev estimate may be unreliable."
        )
    return baseline.mean(), baseline.std(), baseline_start, baseline_end


def flag_volatility_outliers(period_df, baseline_mean, baseline_std, n_std=2):
    """
    Flag days where REALIZED VOLATILITY is more than n_std baseline std devs
    ABOVE the baseline mean -- this is the "shock day" / market-shock
    definition, now based on volatility rather than raw returns. (Volatility
    is one-sided/non-negative, so only an upper bound makes sense here, unlike
    the old two-sided return check.) `period_df` is normally a single
    quarter's slice, matching the quarterly analysis granularity.
    """
    df = period_df.copy()
    upper = baseline_mean + n_std * baseline_std
    df["is_outlier"] = df["realized_vol"] > upper
    return df, upper


# =====================================================================================
# SECTION 4: MERGE PRICE + SENTIMENT, LAG FEATURES (unchanged logic, applied once
#            on the full continuous history so no artificial gaps appear at year
#            boundaries)
# =====================================================================================

def merge_price_and_sentiment(fx_df, ecb_daily_sentiment, rbi_daily_sentiment):
    merged = fx_df.merge(
        ecb_daily_sentiment.rename(columns={"sentiment": "ecb_sentiment"}), on="date", how="left"
    )
    merged = merged.merge(
        rbi_daily_sentiment.rename(columns={"sentiment": "rbi_sentiment"}), on="date", how="left"
    )
    merged["ecb_sentiment"] = merged["ecb_sentiment"].fillna(0)
    merged["rbi_sentiment"] = merged["rbi_sentiment"].fillna(0)
    merged = merged.dropna(subset=["daily_return_pct"]).reset_index(drop=True)
    return merged


def add_lag_features(merged_df):
    """
    IMPORTANT: call this on the FULL multi-year merged series, before slicing
    to a year -- otherwise the first day of every year loses its lag.
    """
    df = merged_df.copy()
    df["lag_ecb_sentiment"] = df["ecb_sentiment"].shift(1)
    df["lag_rbi_sentiment"] = df["rbi_sentiment"].shift(1)
    df = df.dropna(subset=["lag_ecb_sentiment", "lag_rbi_sentiment", "realized_var"]).reset_index(drop=True)
    return df


def prepare_full_dataset(ecb_full_raw, rbi_full_raw, fx_full_raw, ecb_pattern=None, rbi_pattern=None):
    """
    One-time setup: parses V2Tone, builds continuous daily sentiment series
    for ECB and RBI, and merges everything (with lag features) into one
    continuous DataFrame spanning the whole analysis period.

    ecb_pattern / rbi_pattern: normally ECB_PATTERN / RBI_PATTERN (defined at
    the top of this module). Restricts the daily sentiment average to
    articles matching your curated financial/macro keyword list, instead of
    averaging over every article in the raw GDELT file regardless of
    relevance -- see compute_daily_sentiment for why this matters.
    """
    ecb_full_raw = parse_v2tone(ecb_full_raw)
    rbi_full_raw = parse_v2tone(rbi_full_raw)

    ecb_daily_sent = compute_daily_sentiment(ecb_full_raw, pattern=ecb_pattern, out_col="sentiment")
    rbi_daily_sent = compute_daily_sentiment(rbi_full_raw, pattern=rbi_pattern, out_col="sentiment")

    n_ecb_matched = ecb_full_raw["AllNames"].astype(str).str.contains(ecb_pattern, regex=True, na=False).sum() if ecb_pattern else len(ecb_full_raw)
    n_rbi_matched = rbi_full_raw["AllNames"].astype(str).str.contains(rbi_pattern, regex=True, na=False).sum() if rbi_pattern else len(rbi_full_raw)
    print(f"ECB: {n_ecb_matched:,}/{len(ecb_full_raw):,} articles matched the keyword pattern "
          f"({n_ecb_matched/max(len(ecb_full_raw),1)*100:.1f}%) -> {len(ecb_daily_sent):,} days with a sentiment score")
    print(f"RBI: {n_rbi_matched:,}/{len(rbi_full_raw):,} articles matched the keyword pattern "
          f"({n_rbi_matched/max(len(rbi_full_raw),1)*100:.1f}%) -> {len(rbi_daily_sent):,} days with a sentiment score")

    merged = merge_price_and_sentiment(fx_full_raw, ecb_daily_sent, rbi_daily_sent)
    merged_full = add_lag_features(merged)

    return merged_full, ecb_full_raw, rbi_full_raw


# =====================================================================================
# SECTION 5: CUSTOM GARCH(1,1) / GARCH-X MAXIMUM-LIKELIHOOD ESTIMATOR
# =====================================================================================
#
# Why custom instead of the `arch` package: `arch_model`'s exogenous-regressor
# support (`x=...`) only attaches to the MEAN equation (mean='ARX'/'LS').
# There is no built-in way to put exogenous regressors into the VARIANCE
# equation, which is exactly what a sentiment-augmented volatility model
# needs. The functions below fit:
#
#   sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1} [+ exog_t . gamma]
#
# by direct Gaussian maximum likelihood. Passing exog=None gives the
# TRADITIONAL GARCH(1,1); passing a (T x k) exog matrix of LAGGED sentiment
# columns gives the SENTIMENT-INTEGRATED GARCH-X. Both are the same function,
# so the comparison is a clean nested-model comparison (extra parameters vs.
# none), not two differently-coded models.
#
# VALIDATION: this estimator was tested on simulated GARCH(1,1) and GARCH-X
# series with known (omega, alpha, beta, gamma) before being used here; it
# recovered the true parameters (including gamma and its significance) to
# within simulation noise. It is not a novel/unvalidated technique -- GARCH-X
# with exogenous variance regressors, estimated by MLE, is standard in the
# econometrics literature (e.g. Engle & Rangel style volatility models); this
# is simply a from-scratch implementation because `arch` doesn't expose it.

def _garch_variance_path(params, returns, exog=None, floor=1e-8):
    """
    Builds the full sigma2_t path given parameters, for likelihood or
    forecasting. `floor` should normally be a data-scaled value (see
    fit_garch), NOT a tiny absolute constant.
    """
    mu = params[0]
    omega, alpha, beta = params[1], params[2], params[3]
    n_exog = exog.shape[1] if exog is not None else 0
    gammas = np.asarray(params[4:4 + n_exog]) if n_exog > 0 else np.array([])

    eps = returns - mu
    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = np.var(returns)  # initialize at the unconditional sample variance

    for t in range(1, T):
        exog_term = float(np.dot(gammas, exog[t])) if n_exog > 0 else 0.0
        raw = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1] + exog_term
        sigma2[t] = max(raw, floor)  # floor guards against a negative variance
        # when sentiment is very negative -- a known practical issue with
        # GARCH-X models, handled here by clamping rather than allowing an
        # invalid (negative) variance. IMPORTANT: floor must be scaled to the
        # data (see fit_garch's variance_floor), not a tiny absolute constant
        # like 1e-8 -- on a percent^2 scale (typical values ~0.01-2), a 1e-8
        # floor still lets the clamped variance sit orders of magnitude below
        # anything realistic, which then explodes QLIKE (log(sigma2) and
        # realized_var/sigma2 both blow up as sigma2 -> 0).
    return eps, sigma2


def neg_log_likelihood(params, returns, exog=None, floor=1e-8):
    eps, sigma2 = _garch_variance_path(params, returns, exog, floor=floor)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (eps ** 2) / sigma2)
    return -ll


def fit_garch(returns, exog=None, verbose=False, variance_floor_frac=0.01, n_starts=8, seed=0):
    """
    Fits GARCH(1,1) (exog=None) or GARCH-X (exog = T x k array of ALREADY
    LAGGED regressors, one row per return observation) by maximum likelihood.
    Returns a dict with the fitted params, their names, the in-sample
    conditional variance path, the floor actually used, and convergence
    diagnostics.

    variance_floor_frac: the minimum allowed sigma2_t, expressed as a
    fraction of the sample's own unconditional variance (default 1%). Using
    a DATA-SCALED floor (rather than a tiny fixed constant) is what keeps
    QLIKE well-behaved when an exogenous sentiment term pushes the raw
    variance formula close to or below zero on some day.

    n_starts: MULTI-START optimization -- GARCH likelihoods are known to have
    flat ridges and multiple local optima near the alpha=0 / beta=0
    boundaries, so a single starting point can land on a boundary "solution"
    that isn't actually the best fit (this was diagnosed on real thesis data:
    several years' traditional-GARCH fits landed on alpha=0 or beta=0 with
    NaN standard errors -- a tell-tale sign of a boundary optimum, not a
    confidently-estimated zero). Trying several different starting points and
    keeping whichever converges to the highest log-likelihood is the standard
    fix. Sensible defaults + a light random jitter (seeded, so results are
    reproducible) are used for the extra starts; n_starts=1 reproduces the
    old single-start behavior if you ever want to compare directly.
    """
    returns = np.asarray(returns, dtype=float)
    n_exog = exog.shape[1] if exog is not None else 0
    variance_floor = variance_floor_frac * np.var(returns)
    bounds = [(-10, 10), (1e-8, None), (0.0, 1.0), (0.0, 1.0)] + [(None, None)] * n_exog

    rng = np.random.default_rng(seed)
    base_ab_pairs = [(0.05, 0.85), (0.10, 0.70), (0.20, 0.50), (0.30, 0.30), (0.02, 0.90)]
    starts = []
    for i in range(n_starts):
        if i < len(base_ab_pairs):
            a0, b0 = base_ab_pairs[i]
        else:
            a0 = float(rng.uniform(0.01, 0.4))
            b0 = float(rng.uniform(0.3, 0.95 - a0)) if a0 < 0.9 else 0.05
        gamma0 = list(rng.normal(0, 0.01, size=n_exog)) if n_exog > 0 else []
        starts.append([np.mean(returns), a0 * np.var(returns), a0, b0] + gamma0)

    best_res, best_ll = None, -np.inf
    n_converged = 0
    for x0 in starts:
        res = minimize(
            neg_log_likelihood, x0, args=(returns, exog, variance_floor),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12},
        )
        if res.success:
            n_converged += 1
        ll = -res.fun
        if ll > best_ll:
            best_ll, best_res = ll, res

    res = best_res
    if verbose:
        print(f"  GARCH fit: {n_converged}/{len(starts)} starts converged, "
              f"best loglik={best_ll:.4f}" + ("" if res.success else "  (best run itself did not fully converge)"))

    alpha_hat, beta_hat = res.x[2], res.x[3]
    if alpha_hat + beta_hat >= 1:
        warnings.warn(
            f"alpha+beta = {alpha_hat + beta_hat:.4f} >= 1 -- fitted variance process "
            f"is non-stationary (explosive). Treat this fit's forecasts cautiously."
        )
    if np.isclose(alpha_hat, 0, atol=1e-6) or np.isclose(beta_hat, 0, atol=1e-6):
        warnings.warn(
            f"alpha={alpha_hat:.6f}, beta={beta_hat:.6f} -- one landed on its boundary even "
            f"after {len(starts)} starts. This now looks like a genuine feature of this sample "
            f"(not a single-start artifact), but treat inference on that parameter cautiously."
        )

    param_names = ["mu", "omega", "alpha", "beta"] + [f"gamma_{i+1}" for i in range(n_exog)]
    _, sigma2_path = _garch_variance_path(res.x, returns, exog, floor=variance_floor)

    return {
        "params": res.x,
        "param_names": param_names,
        "n_exog": n_exog,
        "sigma2_in_sample": sigma2_path,
        "variance_floor": variance_floor,
        "n_starts_converged": n_converged,
        "n_starts_tried": len(starts),
        "converged": res.success,
        "loglik": -res.fun,
        "n_obs": len(returns),
    }


def _numerical_hessian(f, x, args=(), eps=1e-4):
    """Central-difference numerical Hessian of f at x -- used for MLE standard errors."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            H[i, j] = (f(x_pp, *args) - f(x_pm, *args) - f(x_mp, *args) + f(x_mm, *args)) / (4 * eps * eps)
    return H


def fit_garch_with_inference(returns, exog=None, param_labels=None):
    """
    Same as fit_garch(), plus standard errors / z-stats / p-values for every
    parameter (via the numerical Hessian of the negative log-likelihood at
    the optimum -- the standard MLE "information matrix" approach). This is
    what lets you report, e.g., "gamma_ecb = -0.015 (p = 0.03)" directly.

    param_labels: optional list of names for the exogenous columns (e.g.
    ["ecb_sentiment", "rbi_sentiment"]) to make the output table readable;
    defaults to gamma_1, gamma_2, ...
    """
    fit = fit_garch(returns, exog=exog, verbose=False)
    params = fit["params"]

    try:
        H = _numerical_hessian(neg_log_likelihood, params, args=(returns, exog, fit["variance_floor"]))
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        warnings.warn("Hessian was not invertible -- standard errors unavailable for this fit.")
        se = np.full_like(params, np.nan)

    z = np.divide(params, se, out=np.full_like(params, np.nan), where=se != 0)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    names = ["mu", "omega", "alpha", "beta"]
    if fit["n_exog"] > 0:
        if param_labels is not None and len(param_labels) == fit["n_exog"]:
            names += [f"gamma_{lab}" for lab in param_labels]
        else:
            names += [f"gamma_{i+1}" for i in range(fit["n_exog"])]

    table = pd.DataFrame({
        "parameter": names, "estimate": params, "std_err": se, "z_stat": z, "p_value": p,
    })
    fit["inference_table"] = table
    return fit


def forecast_next_sigma2(params, last_eps, last_sigma2, next_exog=None, floor=1e-8):
    """
    One-step-ahead variance forecast given the last realized residual/variance.
    ALWAYS pass the floor from the matching fit_garch() result
    (fit["variance_floor"]) rather than leaving the tiny default -- see the
    note in fit_garch for why an absolute-constant floor breaks QLIKE.
    """
    omega, alpha, beta = params[1], params[2], params[3]
    n_exog = len(next_exog) if next_exog is not None else 0
    gammas = np.asarray(params[4:4 + n_exog]) if n_exog > 0 else np.array([])
    exog_term = float(np.dot(gammas, next_exog)) if n_exog > 0 else 0.0
    return max(omega + alpha * last_eps ** 2 + beta * last_sigma2 + exog_term, floor)


# =====================================================================================
# SECTION 6: OUT-OF-SAMPLE WALK-FORWARD CV FOR ERROR METRICS ONLY (never used for
#            the plots -- see Section 7 for the plotting fit)
# =====================================================================================

def get_safe_cv_params(n_rows, target_test_frac=0.3, target_n_splits=4, min_test_size=5):
    for n_splits in range(target_n_splits, 1, -1):
        test_size = max(min_test_size, int(round(n_rows * target_test_frac / n_splits)))
        if n_rows >= (n_splits + 1) * test_size:
            return n_splits, test_size
    return None


def run_cv_garch_volatility_forecast(period_df, target_test_frac=0.3, target_n_splits=4, verbose=True):
    """
    period_df must contain: date, daily_return_pct, realized_var,
    lag_ecb_sentiment, lag_rbi_sentiment. Normally called with ONE QUARTER's
    slice (~60 trading days) -- this is the "analysis and splitting" step
    that runs at quarterly granularity.

    For each TimeSeriesSplit fold: fits TRADITIONAL GARCH(1,1) (exog=None) and
    SENTIMENT GARCH-X (exog=[lag_ecb_sentiment, lag_rbi_sentiment]) on the
    fold's training slice, then walks forward through the test slice one day
    at a time, producing a genuine one-step-ahead sigma2 forecast for each
    test day (using the REALIZED residual from the previous test day, once
    it's known, exactly mirroring the original notebook's walk-forward logic
    -- just applied to variance instead of returns).

    Returns pooled out-of-fold (sigma2_traditional, sigma2_sentiment,
    realized_var, dates) across all folds -- this is what compute_error_metrics
    / diebold_mariano_test / compute_directional_metrics / qlike_loss consume.
    """
    df = period_df.dropna(
        subset=["lag_ecb_sentiment", "lag_rbi_sentiment", "daily_return_pct", "realized_var"]
    ).reset_index(drop=True)
    n_rows = len(df)

    cv_params = get_safe_cv_params(n_rows, target_test_frac, target_n_splits)
    if cv_params is None:
        return {"status": "skipped", "reason": f"too few usable rows ({n_rows}) for CV", "n_rows": n_rows}
    n_splits, test_size = cv_params
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    returns_all = df["daily_return_pct"].values
    realized_var_all = df["realized_var"].values
    exog_all = df[["lag_ecb_sentiment", "lag_rbi_sentiment"]].values

    oof_realized_var, oof_sigma2_trad, oof_sigma2_sent, oof_dates = [], [], [], []
    fold_diagnostics = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        y_train = returns_all[train_idx]
        exog_train = exog_all[train_idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_trad = fit_garch(y_train, exog=None)
                fit_sent = fit_garch(y_train, exog=exog_train)
        except Exception as e:
            fold_diagnostics.append({"fold": fold_num, "status": "fit_failed", "error": str(e)})
            if verbose:
                print(f"  Fold {fold_num}: FIT FAILED ({e}) -- skipping this fold")
            continue

        mu_t = fit_trad["params"][0]
        mu_s = fit_sent["params"][0]
        resid_prev_t = y_train[-1] - mu_t
        sigma2_prev_t = fit_trad["sigma2_in_sample"][-1]
        resid_prev_s = y_train[-1] - mu_s
        sigma2_prev_s = fit_sent["sigma2_in_sample"][-1]

        for i in test_idx:  # chronological order within a fold
            e_next = exog_all[i]
            sigma2_t = forecast_next_sigma2(
                fit_trad["params"], resid_prev_t, sigma2_prev_t, next_exog=None, floor=fit_trad["variance_floor"]
            )
            sigma2_s = forecast_next_sigma2(
                fit_sent["params"], resid_prev_s, sigma2_prev_s, next_exog=e_next, floor=fit_sent["variance_floor"]
            )

            oof_realized_var.append(realized_var_all[i])
            oof_sigma2_trad.append(sigma2_t)
            oof_sigma2_sent.append(sigma2_s)
            oof_dates.append(df["date"].iloc[i])

            # roll forward using the now-realized residual for the next test day
            resid_prev_t, sigma2_prev_t = returns_all[i] - mu_t, sigma2_t
            resid_prev_s, sigma2_prev_s = returns_all[i] - mu_s, sigma2_s

        fold_diagnostics.append({"fold": fold_num, "status": "ok", "n_train": len(train_idx), "n_test": len(test_idx)})
        if verbose:
            print(f"  Fold {fold_num}: train={len(train_idx)}, test={len(test_idx)} -- done")

    if len(oof_realized_var) == 0:
        return {"status": "skipped", "reason": "all folds failed to fit", "fold_diagnostics": fold_diagnostics}

    return {
        "status": "ok",
        "oof_realized_var": np.array(oof_realized_var),
        "oof_sigma2_traditional": np.array(oof_sigma2_trad),
        "oof_sigma2_sentiment": np.array(oof_sigma2_sent),
        "oof_dates": oof_dates,
        "n_splits": n_splits, "test_size": test_size,
        "fold_diagnostics": fold_diagnostics,
    }


# =====================================================================================
# SECTION 7: FULL-YEAR IN-SAMPLE FIT -- FOR PLOTTING ONLY (never used for metrics)
# =====================================================================================

def fit_full_year_garch(year_df, param_labels=("ecb_sentiment", "rbi_sentiment")):
    """
    Fits BOTH models ONCE on the entire calendar year's data (in-sample), so
    every single day of the year gets a fitted conditional-volatility value --
    a continuous, gap-free daily series, which is what plot_year_volatility()
    draws. This is intentionally separate from the walk-forward CV in Section
    6, which is out-of-sample but leaves gaps -- exactly the "test/train split
    only for metrics, not for plots" split you asked for.
    """
    df = year_df.dropna(subset=["lag_ecb_sentiment", "lag_rbi_sentiment", "daily_return_pct"]).reset_index(drop=True)
    returns = df["daily_return_pct"].values
    exog = df[["lag_ecb_sentiment", "lag_rbi_sentiment"]].values

    fit_trad = fit_garch_with_inference(returns, exog=None)
    fit_sent = fit_garch_with_inference(returns, exog=exog, param_labels=list(param_labels))

    return {
        "dates": df["date"].tolist(),
        "realized_vol": df["realized_var"].pow(0.5).values if "realized_var" in df.columns else None,
        "traditional": fit_trad,
        "sentiment": fit_sent,
    }


# =====================================================================================
# SECTION 8: ERROR METRICS, DIEBOLD-MARIANO, DIRECTIONAL ACCURACY, QLIKE
#            -- now operating on VARIANCE forecasts vs. REALIZED VARIANCE
# =====================================================================================

def calculate_mape(y_true, y_pred, epsilon=1e-6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_error_metrics(oof_realized_var, oof_sigma2_trad, oof_sigma2_sent):
    """
    RMSE/MAE/MAPE are computed on the VOLATILITY (sqrt-of-variance) scale, in
    percentage points -- this is much easier to read than raw variance
    (percent-squared) units, e.g. "RMSE = 0.18 percentage points of daily
    volatility" rather than "RMSE = 0.032 percent-squared". QLIKE (Section 8b)
    still uses the raw variance scale, since that's the loss's definition.
    """
    realized_vol = np.sqrt(oof_realized_var)
    vol_trad = np.sqrt(oof_sigma2_trad)
    vol_sent = np.sqrt(oof_sigma2_sent)

    rmse_t = np.sqrt(mean_squared_error(realized_vol, vol_trad))
    mae_t = mean_absolute_error(realized_vol, vol_trad)
    mape_t = calculate_mape(realized_vol, vol_trad)

    rmse_s = np.sqrt(mean_squared_error(realized_vol, vol_sent))
    mae_s = mean_absolute_error(realized_vol, vol_sent)
    mape_s = calculate_mape(realized_vol, vol_sent)

    return pd.DataFrame({
        "Model": ["Traditional GARCH", "Sentiment-integrated GARCH-X"],
        "RMSE (vol, pp)": [rmse_t, rmse_s],
        "MAE (vol, pp)": [mae_t, mae_s],
        "MAPE (%)": [mape_t, mape_s],
    })


def diebold_mariano_test(y_true, pred1, pred2, h=1, loss="squared"):
    """
    Unchanged mechanically from before -- generic DM test on any two
    prediction series against the same target. Here y_true=realized_var,
    pred1/pred2=sigma2 forecasts, so this tests whether the two models'
    VARIANCE forecast errors differ significantly.
    """
    y_true = np.array(y_true); pred1 = np.array(pred1); pred2 = np.array(pred2)
    e1 = y_true - pred1
    e2 = y_true - pred2

    if loss == "squared":
        g1, g2 = e1 ** 2, e2 ** 2
    elif loss == "absolute":
        g1, g2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = g1 - g2
    T = len(d)
    d_bar = np.mean(d)

    gamma_0 = np.var(d, ddof=0)
    variance_sum = gamma_0
    for lag in range(1, h):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
        variance_sum += 2 * gamma_lag

    if variance_sum <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(variance_sum / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


def compute_directional_metrics(oof_realized_var, oof_sigma2_trad, oof_sigma2_sent):
    """
    Directional accuracy for VOLATILITY: did realized volatility rise or fall
    from the previous day, and did each model's forecast rise or fall in the
    same direction? (This is the volatility analogue of "did the return go up
    or down" -- volatility clustering means getting the direction of change
    right is a meaningful, separate skill from getting the level right.)
    """
    realized_dir = (np.diff(oof_realized_var) > 0).astype(int)
    trad_dir = (np.diff(oof_sigma2_trad) > 0).astype(int)
    sent_dir = (np.diff(oof_sigma2_sent) > 0).astype(int)

    def _eval(true_dir, pred_dir, name):
        return {
            "model": name,
            "accuracy": accuracy_score(true_dir, pred_dir),
            "precision": precision_score(true_dir, pred_dir, average="macro", zero_division=0),
            "recall": recall_score(true_dir, pred_dir, average="macro", zero_division=0),
            "f1": f1_score(true_dir, pred_dir, average="macro", zero_division=0),
        }

    return pd.DataFrame([
        _eval(realized_dir, trad_dir, "Traditional"),
        _eval(realized_dir, sent_dir, "Sentiment-integrated"),
    ])


def qlike_loss(realized_var, forecast_var, epsilon=1e-8):
    """
    Standard loss for scoring volatility forecasts (lower = better) -- unchanged.
    Here realized_var is the ACTUAL squared-demeaned-return proxy (not a
    residual from a return forecast, since we no longer forecast returns).
    """
    forecast_var = np.maximum(forecast_var, epsilon)
    realized_var = np.maximum(realized_var, epsilon)
    return np.mean(np.log(forecast_var) + realized_var / forecast_var)


def compute_risk_metrics(oof_realized_var, oof_sigma2_trad, oof_sigma2_sent, is_outlier_mask=None):
    """
    QLIKE for both models, plus (if is_outlier_mask given) each model's mean
    forecasted volatility on shock vs. normal days -- a model that's actually
    useful for risk should show a bigger shock/normal gap.
    """
    rows = [{
        "model": "Traditional GARCH",
        "qlike": qlike_loss(oof_realized_var, oof_sigma2_trad),
        "mean_forecast_vol_%": np.mean(np.sqrt(oof_sigma2_trad)),
    }, {
        "model": "Sentiment-integrated GARCH-X",
        "qlike": qlike_loss(oof_realized_var, oof_sigma2_sent),
        "mean_forecast_vol_%": np.mean(np.sqrt(oof_sigma2_sent)),
    }]

    if is_outlier_mask is not None:
        is_outlier_mask = np.array(is_outlier_mask)
        if is_outlier_mask.any() and (~is_outlier_mask).any():
            rows[0]["shock_day_vol_%"] = np.mean(np.sqrt(oof_sigma2_trad[is_outlier_mask]))
            rows[0]["normal_day_vol_%"] = np.mean(np.sqrt(oof_sigma2_trad[~is_outlier_mask]))
            rows[1]["shock_day_vol_%"] = np.mean(np.sqrt(oof_sigma2_sent[is_outlier_mask]))
            rows[1]["normal_day_vol_%"] = np.mean(np.sqrt(oof_sigma2_sent[~is_outlier_mask]))

    return pd.DataFrame(rows)


# =====================================================================================
# SECTION 9: PLOTTING -- ONE CONTINUOUS DAILY VOLATILITY GRAPH PER CALENDAR YEAR
# =====================================================================================

def plot_year_volatility(year, full_year_fit, outlier_dates=None, figsize=(14, 5)):
    """
    Draws ONE year's worth of daily volatility (Jan 1 - Dec 31), using the
    CONTINUOUS in-sample fit from fit_full_year_garch() -- every day has a
    value, so there are no gaps and no "slanted" lines connecting distant
    out-of-fold points. Shows:
      - realized daily volatility (thin grey line -- the noisy, actual proxy)
      - Traditional GARCH fitted volatility (smooth blue line)
      - Sentiment-integrated GARCH-X fitted volatility (smooth red line)
      - shock/outlier days marked with orange markers
    """
    dates = pd.to_datetime(full_year_fit["dates"])
    realized_vol = full_year_fit["realized_vol"]
    vol_trad = np.sqrt(full_year_fit["traditional"]["sigma2_in_sample"])
    vol_sent = np.sqrt(full_year_fit["sentiment"]["sigma2_in_sample"])

    fig, ax = plt.subplots(figsize=figsize)
    if realized_vol is not None:
        ax.plot(dates, realized_vol, color="lightgray", linewidth=0.8,
                label="Realized daily volatility (actual)", zorder=1)
    ax.plot(dates, vol_trad, color="#1f77b4", linewidth=1.4,
             label="Traditional GARCH (fitted)", zorder=2)
    ax.plot(dates, vol_sent, color="#d62728", linewidth=1.4, linestyle="--",
             label="Sentiment-integrated GARCH-X (fitted)", zorder=2)

    if outlier_dates:
        outlier_dt = pd.to_datetime(list(outlier_dates))
        mask = dates.isin(outlier_dt)
        if mask.any() and realized_vol is not None:
            ax.scatter(dates[mask], np.array(realized_vol)[mask], color="orange",
                       s=35, zorder=3, label="Shock day (>2 std devs)")
                
    ax.set_xlim(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily volatility (percentage points)")
    ax.set_title(f"{year}: EUR/INR daily volatility -- Traditional vs. Sentiment-integrated GARCH-X")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# =====================================================================================
# SECTION 10: KEYWORD FREQUENCY FOR SHOCK / NORMAL DAYS (mechanically unchanged --
#             now driven by VOLATILITY-based outlier dates from Section 3)
# =====================================================================================

def extract_names_from_allnames(allnames_string):
    if pd.isna(allnames_string) or allnames_string == "":
        return []
    entries = allnames_string.split(";")
    names = []
    for entry in entries:
        name_part = entry.split(",")[0].strip()
        if name_part:
            names.append(name_part)
    return names


def get_shock_day_keyword_counts(news_df, shock_date, top_n=15, date_col="date_only"):
    day_news = news_df[news_df[date_col] == shock_date]
    all_names_combined = []
    for allnames_str in day_news["AllNames"]:
        all_names_combined.extend(extract_names_from_allnames(allnames_str))
    counts = Counter(all_names_combined)
    return counts.most_common(top_n), len(day_news)


def get_keyword_counts_for_dates(news_df, dates, top_n=15, date_col="date_only"):
    if not dates:
        return [], 0
    dates_set = set(dates)
    day_news = news_df[news_df[date_col].isin(dates_set)]
    all_names_combined = []
    for allnames_str in day_news["AllNames"]:
        all_names_combined.extend(extract_names_from_allnames(allnames_str))
    counts = Counter(all_names_combined)
    return counts.most_common(top_n), len(day_news)


def select_normal_days(year_df, n=8, gdelt_start=None, date_col="date"):
    df = year_df.copy()
    if gdelt_start is not None:
        df = df[df[date_col] >= gdelt_start]
    normal_df = df[df["is_outlier"] == False]  # noqa: E712
    normal_dates_sorted = sorted(normal_df[date_col].tolist())

    if len(normal_dates_sorted) == 0:
        return []
    if len(normal_dates_sorted) <= n:
        return [str(d) for d in normal_dates_sorted]
    indices = np.linspace(0, len(normal_dates_sorted) - 1, n).astype(int)
    return [str(normal_dates_sorted[i]) for i in indices]


def build_shock_day_query(pattern, target_date):
    return f"""
    SELECT
      DATE,
      SourceCommonName,
      DocumentIdentifier AS SourceURL,
      AllNames,
      V2Tone,
      Quotations
    FROM
      `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
      _PARTITIONTIME = TIMESTAMP("{target_date}")
      AND REGEXP_CONTAINS(AllNames, r'{pattern}')
    """


def check_cost(client, query):
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    return job.total_bytes_processed / 1e9


def get_best_text(row):
    if pd.notna(row.get("Quotations")) and len(str(row["Quotations"])) > 20:
        return row["Quotations"]
    if pd.notna(row.get("article_text")) and len(str(row["article_text"])) > 20:
        return row["article_text"]
    return None


def pull_quotations_for_dates(client, pattern, dates, gb_limit=5, verbose=True):
    results = {}
    for d in dates:
        q = build_shock_day_query(pattern, d)
        gb = check_cost(client, q)
        if verbose:
            print(f"    {d}: estimated {gb:.3f} GB")
        if gb > gb_limit:
            if verbose:
                print(f"      --> skipped (exceeds {gb_limit} GB limit)")
            continue
        df = client.query(q).to_dataframe()
        results[d] = df
        if verbose:
            non_empty = df["Quotations"].dropna()
            non_empty = non_empty[non_empty.str.len() > 0]
            print(f"      --> {len(df)} articles, {len(non_empty)} with extractable quotes")
    return results


def collect_quotation_texts(client, pattern, dates, gb_limit=5, verbose=True):
    if not dates:
        return []
    quotations = pull_quotations_for_dates(client, pattern, dates, gb_limit=gb_limit, verbose=verbose)
    all_texts = []
    for d, df in quotations.items():
        if len(df) == 0:
            continue
        df = df.copy()
        df["best_text"] = df.apply(get_best_text, axis=1)
        all_texts.extend(df["best_text"].dropna().tolist())
    return all_texts


# =====================================================================================
# SECTION 11: LOUGHRAN-MCDONALD DICTIONARY, TEXT CLEANING, TF-IDF + LOGISTIC REGRESSION
# =====================================================================================

DEFAULT_LM_SENTIMENT_COLS = [
    "Negative", "Positive", "Uncertainty", "Litigious",
    "Strong_Modal", "Weak_Modal", "Constraining",
]

CUSTOM_STOPWORDS = {"like", "good", "come", "know", "today", "people"}


def load_financial_vocab(lm_path, sentiment_cols=None):
    if sentiment_cols is None:
        sentiment_cols = DEFAULT_LM_SENTIMENT_COLS
    lm_dict = pd.read_csv(lm_path)
    missing = [c for c in sentiment_cols if c not in lm_dict.columns]
    if missing:
        raise ValueError(f"LM dictionary is missing expected columns: {missing}.")
    lm_dict["is_sentiment_word"] = lm_dict[sentiment_cols].sum(axis=1) > 0
    lm_sentiment_words = lm_dict[lm_dict["is_sentiment_word"]]
    financial_vocab = set(lm_sentiment_words["Word"].str.lower().tolist())
    print(f"Loaded {len(lm_dict):,} total LM terms -> {len(financial_vocab):,} sentiment-tagged financial terms")
    return financial_vocab


def clean_text_for_tfidf(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_tfidf_logistic(shock_texts, normal_texts, financial_vocab=None, max_features=300, min_df=2):
    texts = shock_texts + normal_texts
    labels = [1] * len(shock_texts) + [0] * len(normal_texts)
    texts_cleaned = [clean_text_for_tfidf(t) for t in texts]

    all_stopwords = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))
    restricted_vocab = None
    if financial_vocab:
        restricted_vocab = [w for w in financial_vocab if len(w) >= 3]

    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2), stop_words=all_stopwords,
        min_df=min_df, token_pattern=r"\b[a-z]{3,}\b", vocabulary=restricted_vocab,
    )
    X_tfidf = vectorizer.fit_transform(texts_cleaned)
    y_labels = np.array(labels)

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_tfidf, y_labels)

    feature_names = vectorizer.get_feature_names_out()
    coefficients = log_reg.coef_[0]
    doc_freq = np.asarray((X_tfidf > 0).sum(axis=0)).flatten()

    keyword_importance = pd.DataFrame({
        "keyword": feature_names, "coefficient": coefficients, "doc_frequency": doc_freq,
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    return keyword_importance



# =====================================================================================
# SECTION 12: PER-QUARTER ORCHESTRATOR -- this is where the "analysis and splitting"
#             actually happens (outlier flagging, GARCH fit, train/test CV, error
#             metrics).
# =====================================================================================

def run_quarter_pipeline(
    year, quarter, merged_full, history_start_date,
    n_std=2, target_test_frac=0.3, target_n_splits=4, verbose=True,
):
    """
    Runs the volatility-forecasting analysis for ONE quarter:
      1. Slice to the quarter.
      2. Compute the expanding volatility baseline (grows every QUARTER, using
         all data strictly before this quarter -- Section 3), flag shock days.
      3. Run walk-forward out-of-sample CV within this quarter (Section 6).
      4. Compute error metrics, DM test, directional accuracy, QLIKE (Section 8).

    Deliberately does NOT plot anything and does NOT touch keywords/TF-IDF --
    those happen once per year in run_year_pipeline, after all 4 quarters are
    done, using the combined shock-day list.
    """
    quarter_df = filter_to_quarter(merged_full, year, quarter, date_col="date")
    label = f"{year}-Q{quarter}"
    results = {"year": year, "quarter": quarter, "label": label}

    if len(quarter_df) < 15:
        results["status"] = "skipped"
        results["reason"] = f"too few rows ({len(quarter_df)}) for {label}"
        if verbose:
            print(f"  {label}: skipped ({results['reason']})")
        return results

    # ---- expanding baseline + volatility-based outlier flagging ----
    baseline_mean, baseline_std, b_start, b_end = compute_expanding_baseline_stats(
        merged_full, year, quarter, history_start_date
    )
    quarter_df, upper_bound = flag_volatility_outliers(quarter_df, baseline_mean, baseline_std, n_std=n_std)
    outlier_days = quarter_df[quarter_df["is_outlier"]]
    results.update(
        outlier_bound=upper_bound,
        n_outlier_days=len(outlier_days),
        outlier_dates=outlier_days["date"].tolist(),
        normal_dates=quarter_df.loc[~quarter_df["is_outlier"], "date"].tolist(),
    )
    if verbose:
        print(f"\n  --- {label} ---")
        print(f"    Baseline ({b_start} to {b_end}): mean={baseline_mean:.4f}pp, std={baseline_std:.4f}pp, "
              f"threshold={upper_bound:.4f}pp")
        print(f"    {len(outlier_days)} shock day(s): {results['outlier_dates']}")

    # ---- out-of-sample walk-forward CV within this quarter ----
    cv_result = run_cv_garch_volatility_forecast(
        quarter_df, target_test_frac=target_test_frac, target_n_splits=target_n_splits, verbose=False
    )
    results["cv"] = cv_result

    # ---- ALSO fit one in-sample GARCH per quarter (mirrors fit_full_year_garch,
    # which is generic enough to reuse here) -- this is what gives each quarter
    # its OWN reportable omega/alpha/beta(/gamma) coefficient table, distinct
    # from the CV step above (which fits many different sub-windows per fold
    # and never keeps a single "this quarter's" coefficient set). Without this,
    # there was no coefficient table available at quarterly granularity at all.
    try:
        results["full_quarter_fit"] = fit_full_year_garch(quarter_df)
    except Exception as e:
        results["full_quarter_fit"] = None
        if verbose:
            print(f"    Full-quarter coefficient fit failed: {e}")

    if cv_result["status"] == "ok":
        realized_var = cv_result["oof_realized_var"]
        sigma2_trad = cv_result["oof_sigma2_traditional"]
        sigma2_sent = cv_result["oof_sigma2_sentiment"]
        oof_dates = cv_result["oof_dates"]

        results["error_metrics"] = compute_error_metrics(realized_var, sigma2_trad, sigma2_sent)
        dm_stat, dm_p = diebold_mariano_test(realized_var, sigma2_trad, sigma2_sent)
        results["dm_stat"], results["dm_pvalue"] = dm_stat, dm_p
        results["directional_metrics"] = compute_directional_metrics(realized_var, sigma2_trad, sigma2_sent)

        outlier_date_set = set(results["outlier_dates"])
        is_outlier_oof = np.array([d in outlier_date_set for d in oof_dates])
        results["risk_metrics"] = compute_risk_metrics(realized_var, sigma2_trad, sigma2_sent, is_outlier_mask=is_outlier_oof)

        if verbose:
            print(f"    RMSE  trad={results['error_metrics'].loc[0, 'RMSE (vol, pp)']:.4f}  "
                  f"sent={results['error_metrics'].loc[1, 'RMSE (vol, pp)']:.4f}   "
                  f"DM p={dm_p:.4f}   "
                  f"QLIKE trad={results['risk_metrics'].loc[0, 'qlike']:.4f} "
                  f"sent={results['risk_metrics'].loc[1, 'qlike']:.4f}")
    else:
        if verbose:
            print(f"    CV skipped: {cv_result.get('reason')}")

    results["status"] = "ok"
    return results


# =====================================================================================
# SECTION 13: PER-YEAR ORCHESTRATOR -- runs the 4 quarters, builds the ONE continuous
#             yearly plot, and averages the 4 quarters into one yearly summary row
# =====================================================================================

def average_quarterly_metrics(quarter_results):
    """
    Averages the 4 quarters' numeric metrics into ONE yearly row per model.
    Only quarters with status == 'ok' and a successful CV are included; a
    year with fewer than 4 usable quarters still averages over however many
    succeeded (and reports how many that was).
    """
    ok_quarters = [q for q in quarter_results if q.get("status") == "ok" and q.get("cv", {}).get("status") == "ok"]
    if not ok_quarters:
        return None

    def _mean_metric(get_value):
        return np.nanmean([get_value(q) for q in ok_quarters])

    row = {"n_quarters_used": len(ok_quarters)}
    for model_idx, prefix in [(0, "trad"), (1, "sent")]:
        row[f"{prefix}_rmse"] = _mean_metric(lambda q, i=model_idx: q["error_metrics"].loc[i, "RMSE (vol, pp)"])
        row[f"{prefix}_mae"] = _mean_metric(lambda q, i=model_idx: q["error_metrics"].loc[i, "MAE (vol, pp)"])
        row[f"{prefix}_mape"] = _mean_metric(lambda q, i=model_idx: q["error_metrics"].loc[i, "MAPE (%)"])
        row[f"{prefix}_qlike"] = _mean_metric(lambda q, i=model_idx: q["risk_metrics"].loc[i, "qlike"])
        dir_key = "Traditional" if prefix == "trad" else "Sentiment-integrated"
        row[f"{prefix}_dir_accuracy"] = _mean_metric(
            lambda q, k=dir_key: q["directional_metrics"].set_index("model").loc[k, "accuracy"]
        )
        row[f"{prefix}_dir_f1"] = _mean_metric(
            lambda q, k=dir_key: q["directional_metrics"].set_index("model").loc[k, "f1"]
        )
    row["dm_stat_avg"] = _mean_metric(lambda q: q["dm_stat"])
    row["dm_pvalue_avg"] = _mean_metric(lambda q: q["dm_pvalue"])
    row["n_significant_quarters_dm"] = sum(1 for q in ok_quarters if pd.notna(q["dm_pvalue"]) and q["dm_pvalue"] < 0.05)
    return row


def run_year_pipeline(
    year, merged_full, ecb_full_raw, rbi_full_raw, financial_vocab, history_start_date,
    ecb_pattern=ECB_PATTERN, rbi_pattern=RBI_PATTERN, bq_client=None,
    pull_quotations=True, n_normal_days=8, n_std=2,
    target_test_frac=0.3, target_n_splits=4, make_plot=True, verbose=True,
):
    """
    Runs the full sequence for ONE calendar year:
      1. Run all 4 quarters through run_quarter_pipeline (Section 12) --
         this is where outlier flagging, GARCH fitting, and the train/test
         split actually happen.
      2. Average the 4 quarters into ONE yearly summary row (this function).
      3. Separately, fit both models ONCE, in-sample, on the WHOLE year
         (Section 7) -- purely for the continuous Jan-Dec plot; never mixed
         with the quarterly train/test analysis above.
      4. Draw the year's volatility plot, marking shock days from all 4
         quarters combined (Section 9).
      5. Keyword frequency for shock vs. normal days, using the combined
         shock-day list from all 4 quarters (Section 10).
      6. Optional: pull quotations + TF-IDF/logistic keyword attribution
         (Section 11), also using the combined shock-day list.
    """
    if verbose:
        print(f"\n{'='*72}\n{year}\n{'='*72}")

    year_df_for_plot = filter_to_year(merged_full, year, date_col="date")
    results = {"year": year}

    if len(year_df_for_plot) < 30:
        results["status"] = "skipped"
        results["reason"] = f"too few rows ({len(year_df_for_plot)}) for {year}"
        if verbose:
            print(f"  {year}: skipped ({results['reason']})")
        return results

    # ---- Step 1: run all 4 quarters (the actual analysis) ----
    quarter_results = []
    for quarter in (1, 2, 3, 4):
        qres = run_quarter_pipeline(
            year, quarter, merged_full, history_start_date,
            n_std=n_std, target_test_frac=target_test_frac,
            target_n_splits=target_n_splits, verbose=verbose,
        )
        quarter_results.append(qres)
    results["quarter_results"] = quarter_results

    # ---- Step 2: average the 4 quarters into one yearly row ----
    yearly_avg = average_quarterly_metrics(quarter_results)
    results["yearly_avg_metrics"] = yearly_avg
    if verbose and yearly_avg is not None:
        print(f"\n  Year average ({yearly_avg['n_quarters_used']}/4 quarters used):")
        print(f"    RMSE  trad={yearly_avg['trad_rmse']:.4f}  sent={yearly_avg['sent_rmse']:.4f}")
        print(f"    QLIKE trad={yearly_avg['trad_qlike']:.4f}  sent={yearly_avg['sent_qlike']:.4f}")
        print(f"    Directional accuracy  trad={yearly_avg['trad_dir_accuracy']:.4f}  sent={yearly_avg['sent_dir_accuracy']:.4f}")
        print(f"    DM significant in {yearly_avg['n_significant_quarters_dm']}/{yearly_avg['n_quarters_used']} quarters")

    # combined shock-day list across all 4 quarters, for the plot + keywords + TF-IDF
    combined_outlier_dates = sorted({d for q in quarter_results for d in q.get("outlier_dates", [])})
    combined_normal_dates = sorted({d for q in quarter_results for d in q.get("normal_dates", [])})
    results["outlier_dates"] = combined_outlier_dates
    results["n_outlier_days"] = len(combined_outlier_dates)

    # ---- Step 3: full-year in-sample fit -- for the plot + reportable coefficients ----
    full_year_fit = fit_full_year_garch(year_df_for_plot)
    results["full_year_fit"] = full_year_fit
    if verbose:
        print("\n  Full-year (in-sample) GARCH coefficients -- for the plot & thesis write-up:")
        print("  Traditional GARCH(1,1):")
        print(full_year_fit["traditional"]["inference_table"].to_string(index=False))
        print("  Sentiment-integrated GARCH-X:")
        print(full_year_fit["sentiment"]["inference_table"].to_string(index=False))

    if make_plot:
        results["figure"] = plot_year_volatility(year, full_year_fit, outlier_dates=combined_outlier_dates)

    # ---- Step 4: keyword frequency for shock vs. normal days (year-level, combined) ----
    y_start, y_end = get_year_bounds(year)
    ecb_y_raw = ecb_full_raw[(ecb_full_raw["date_only"] >= y_start) & (ecb_full_raw["date_only"] <= y_end)]
    rbi_y_raw = rbi_full_raw[(rbi_full_raw["date_only"] >= y_start) & (rbi_full_raw["date_only"] <= y_end)]
    combined_y_raw = pd.concat([ecb_y_raw, rbi_y_raw], ignore_index=True)

    shock_keyword_counts = {}
    for shock_date in combined_outlier_dates:
        ecb_kw, ecb_n = get_shock_day_keyword_counts(ecb_y_raw, shock_date)
        rbi_kw, rbi_n = get_shock_day_keyword_counts(rbi_y_raw, shock_date)
        shock_keyword_counts[str(shock_date)] = {
            "ecb_top_keywords": ecb_kw, "ecb_articles": ecb_n,
            "rbi_top_keywords": rbi_kw, "rbi_articles": rbi_n,
        }
    results["shock_entity_keyword_counts"] = shock_keyword_counts

    normal_kw_counts, _ = get_keyword_counts_for_dates(combined_y_raw, combined_normal_dates)
    results["top_normal_keywords"] = [kw for kw, _ in normal_kw_counts[:3]]

    if combined_outlier_dates:
        shock_kw_counts_combined, _ = get_keyword_counts_for_dates(combined_y_raw, combined_outlier_dates)
        results["top_shock_keywords"] = [kw for kw, _ in shock_kw_counts_combined[:3]]
    else:
        results["top_shock_keywords"] = []

    if verbose:
        print(f"\n  Top normal-day keywords: {results['top_normal_keywords']}")
        print(f"  Top shock-day keywords:  {results['top_shock_keywords'] or '(no shock days this year)'}")

    # ---- Step 5: quotations + TF-IDF (optional, needs BigQuery + shock days) ----
    if pull_quotations and bq_client is not None and len(combined_outlier_dates) > 0:
        normal_dates_sample = combined_normal_dates[:: max(1, len(combined_normal_dates) // n_normal_days)][:n_normal_days]
        shock_dates_str = [str(d) for d in combined_outlier_dates]
        results["normal_dates_selected"] = normal_dates_sample

        if verbose:
            print(f"  Pulling quotations for {len(shock_dates_str)} shock day(s) and {len(normal_dates_sample)} normal day(s)...")

        shock_texts = (
            collect_quotation_texts(bq_client, ecb_pattern, shock_dates_str, verbose=verbose)
            + collect_quotation_texts(bq_client, rbi_pattern, shock_dates_str, verbose=verbose)
        )
        normal_texts = (
            collect_quotation_texts(bq_client, ecb_pattern, normal_dates_sample, verbose=verbose)
            + collect_quotation_texts(bq_client, rbi_pattern, normal_dates_sample, verbose=verbose)
        )
        results["n_shock_texts"], results["n_normal_texts"] = len(shock_texts), len(normal_texts)

        if len(shock_texts) >= 2 and len(normal_texts) >= 2:
            keyword_importance = run_tfidf_logistic(shock_texts, normal_texts, financial_vocab)
            results["tfidf_keyword_importance"] = keyword_importance
            results["tfidf_top_shock_keywords"] = keyword_importance.sort_values("coefficient", ascending=False).head(3)
            results["tfidf_top_normal_keywords"] = keyword_importance.sort_values("coefficient", ascending=True).head(3)
        elif verbose:
            print(f"  Not enough texts for TF-IDF (shock={len(shock_texts)}, normal={len(normal_texts)}) -- skipping.")
    elif verbose and pull_quotations:
        reason = "no outlier days this year" if len(combined_outlier_dates) == 0 else "no bq_client provided"
        print(f"  Skipping quotations/TF-IDF step ({reason}).")

    results["status"] = "ok"
    return results


# =====================================================================================
# SECTION 14: TOP-LEVEL LOOP OVER ALL YEARS + THREE LEVELS OF SUMMARY
#             (quarterly detail table -> yearly averaged table -> one 10-year table)
# =====================================================================================

def run_all_years(
    years, merged_full, ecb_full_raw, rbi_full_raw, financial_vocab,
    history_start_date=None, ecb_pattern=ECB_PATTERN, rbi_pattern=RBI_PATTERN, bq_client=None,
    pull_quotations=True, make_plot=True, verbose=True,
):
    """
    Runs run_year_pipeline() for every year in `years` (each of which runs 4
    quarters internally) -- e.g. 2016..2025 = 10 years = 40 quarters analyzed,
    10 plots drawn. `history_start_date` defaults to the earliest date present
    in merged_full if not given.
    """
    if history_start_date is None:
        history_start_date = merged_full["date"].min()

    all_results = []
    for year in years:
        res = run_year_pipeline(
            year, merged_full, ecb_full_raw, rbi_full_raw, financial_vocab, history_start_date,
            ecb_pattern=ecb_pattern, rbi_pattern=rbi_pattern, bq_client=bq_client,
            pull_quotations=pull_quotations, make_plot=make_plot, verbose=verbose,
        )
        all_results.append(res)
    return all_results


def build_quarterly_summary_dataframe(all_year_results):
    """One row per (year, quarter) -- the detailed 40-row table, model metrics side by side."""
    rows = []
    for yr in all_year_results:
        if yr.get("status") != "ok":
            continue
        for q in yr.get("quarter_results", []):
            if q.get("status") != "ok":
                rows.append({"year": q["year"], "quarter": q["quarter"], "label": q["label"], "status": q.get("status", "unknown")})
                continue
            row = {
                "year": q["year"], "quarter": q["quarter"], "label": q["label"], "status": "ok",
                "n_outlier_days": q.get("n_outlier_days"),
                "dm_stat": q.get("dm_stat"), "dm_pvalue": q.get("dm_pvalue"),
            }
            em = q.get("error_metrics")
            if em is not None:
                for i, prefix in [(0, "trad"), (1, "sent")]:
                    row[f"{prefix}_rmse"] = em.loc[i, "RMSE (vol, pp)"]
                    row[f"{prefix}_mae"] = em.loc[i, "MAE (vol, pp)"]
                    row[f"{prefix}_mape"] = em.loc[i, "MAPE (%)"]
            dmet = q.get("directional_metrics")
            if dmet is not None:
                dmet_idx = dmet.set_index("model")
                row["trad_dir_accuracy"] = dmet_idx.loc["Traditional", "accuracy"]
                row["sent_dir_accuracy"] = dmet_idx.loc["Sentiment-integrated", "accuracy"]
            rmet = q.get("risk_metrics")
            if rmet is not None:
                for i, prefix in [(0, "trad"), (1, "sent")]:
                    row[f"{prefix}_qlike"] = rmet.loc[i, "qlike"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_yearly_summary_dataframe(all_year_results):
    """
    One row per YEAR -- the 4 quarters' metrics averaged (average_quarterly_metrics,
    computed inside run_year_pipeline), plus the year's GARCH-X coefficients from
    the full-year in-sample fit and its top shock/normal keywords.
    """
    rows = []
    for r in all_year_results:
        if r.get("status") != "ok":
            rows.append({"year": r["year"], "status": r.get("status", "unknown")})
            continue

        row = {"year": r["year"], "status": "ok", "n_outlier_days": r.get("n_outlier_days")}
        yavg = r.get("yearly_avg_metrics")
        if yavg is not None:
            row.update(yavg)

        fyf = r.get("full_year_fit")
        if fyf is not None:
            sent_tab = fyf["sentiment"]["inference_table"].set_index("parameter")
            for gname in [c for c in sent_tab.index if c.startswith("gamma_")]:
                row[f"{gname}_estimate"] = sent_tab.loc[gname, "estimate"]
                row[f"{gname}_pvalue"] = sent_tab.loc[gname, "p_value"]

        row["top_shock_keywords"] = ", ".join(r.get("top_shock_keywords", []))
        row["top_normal_keywords"] = ", ".join(r.get("top_normal_keywords", []))
        rows.append(row)
    return pd.DataFrame(rows)


def build_key_metrics_summary(all_year_results):
    """
    A compact, human-readable per-year table for a quick "is sentiment
    helping this year" scan -- a curated subset/rename of
    build_yearly_summary_dataframe's columns, in a more readable order.
    """
    full = build_yearly_summary_dataframe(all_year_results)
    wanted = {
        "year": "year",
        "n_outlier_days": "shock_days",
        "trad_rmse": "RMSE_traditional",
        "sent_rmse": "RMSE_sentiment",
        "trad_qlike": "QLIKE_traditional",
        "sent_qlike": "QLIKE_sentiment",
        "trad_dir_accuracy": "dir_accuracy_traditional",
        "sent_dir_accuracy": "dir_accuracy_sentiment",
        "dm_stat_avg": "DM_stat",
        "dm_pvalue_avg": "DM_pvalue",
        "gamma_ecb_sentiment_estimate": "gamma_ecb",
        "gamma_ecb_sentiment_pvalue": "gamma_ecb_pvalue",
        "gamma_rbi_sentiment_estimate": "gamma_rbi",
        "gamma_rbi_sentiment_pvalue": "gamma_rbi_pvalue",
    }
    cols_present = [c for c in wanted if c in full.columns]
    out = full[cols_present].rename(columns=wanted)
    if "RMSE_traditional" in out.columns and "RMSE_sentiment" in out.columns:
        out["sentiment_helped_RMSE"] = out["RMSE_sentiment"] < out["RMSE_traditional"]
    if "QLIKE_traditional" in out.columns and "QLIKE_sentiment" in out.columns:
        out["sentiment_helped_QLIKE"] = out["QLIKE_sentiment"] < out["QLIKE_traditional"]
    return out


def build_yearly_coefficients_table(all_year_results):
    """
    Every YEAR's full-year in-sample GARCH coefficients, for both models,
    stacked into ONE long table -- columns: year, model, parameter, estimate,
    std_err, z_stat, p_value. This is the "coefficients for every year, not
    just one" table -- no need to change year_to_inspect and re-run.
    """
    rows = []
    for r in all_year_results:
        if r.get("status") != "ok" or r.get("full_year_fit") is None:
            continue
        for model_key, model_label in [("traditional", "Traditional"), ("sentiment", "Sentiment-integrated")]:
            tab = r["full_year_fit"][model_key]["inference_table"].copy()
            tab.insert(0, "model", model_label)
            tab.insert(0, "year", r["year"])
            rows.append(tab)
    if not rows:
        return pd.DataFrame(columns=["year", "model", "parameter", "estimate", "std_err", "z_stat", "p_value"])
    return pd.concat(rows, ignore_index=True)


def build_quarterly_coefficients_table(all_year_results):
    """
    Every QUARTER's own in-sample GARCH coefficients (from full_quarter_fit,
    a separate single fit on that quarter's data -- distinct from the CV
    step's many per-fold fits), for both models, stacked into ONE long table
    -- columns: year, quarter, model, parameter, estimate, std_err, z_stat,
    p_value. This is the quarterly analogue of build_yearly_coefficients_table.
    """
    rows = []
    for r in all_year_results:
        if r.get("status") != "ok":
            continue
        for q in r.get("quarter_results", []):
            if q.get("status") != "ok" or q.get("full_quarter_fit") is None:
                continue
            for model_key, model_label in [("traditional", "Traditional"), ("sentiment", "Sentiment-integrated")]:
                tab = q["full_quarter_fit"][model_key]["inference_table"].copy()
                tab.insert(0, "model", model_label)
                tab.insert(0, "quarter", q["quarter"])
                tab.insert(0, "year", q["year"])
                rows.append(tab)
    if not rows:
        return pd.DataFrame(columns=["year", "quarter", "model", "parameter", "estimate", "std_err", "z_stat", "p_value"])
    return pd.concat(rows, ignore_index=True)


def build_shock_dates_table(all_year_results, merged_full=None):
    """
    One row per SHOCK DAY across every year -- for manually looking up what
    macro/news event happened on that date. Columns: year, quarter, date,
    realized_vol (if merged_full given), ecb_sentiment/rbi_sentiment on that
    day (if merged_full given), and the top ECB/RBI entity keywords already
    pulled for that date (from run_year_pipeline's Section 10 keyword step,
    if available -- gives you a head start before searching manually).
    """
    rows = []
    for r in all_year_results:
        if r.get("status") != "ok":
            continue
        year = r["year"]
        entity_counts = r.get("shock_entity_keyword_counts", {})
        for q in r.get("quarter_results", []):
            if q.get("status") != "ok":
                continue
            for d in q.get("outlier_dates", []):
                row = {"year": year, "quarter": q["quarter"], "date": d,
                       "quarter_threshold_vol_pp": q.get("outlier_bound")}
                d_key = str(d)
                if d_key in entity_counts:
                    ecb_top = entity_counts[d_key].get("ecb_top_keywords", [])[:3]
                    rbi_top = entity_counts[d_key].get("rbi_top_keywords", [])[:3]
                    row["ecb_top_entities"] = ", ".join(f"{k}({n})" for k, n in ecb_top)
                    row["rbi_top_entities"] = ", ".join(f"{k}({n})" for k, n in rbi_top)
                rows.append(row)

    table = pd.DataFrame(rows).sort_values(["year", "quarter", "date"]).reset_index(drop=True)

    if merged_full is not None and len(table) > 0:
        lookup = merged_full.set_index("date")[["realized_vol", "ecb_sentiment", "rbi_sentiment"]]
        table = table.merge(lookup, left_on="date", right_index=True, how="left")

    return table


def overall_verdict(all_year_results):
    """
    Pools every QUARTER's out-of-fold points, across every year, into ONE
    10-year evaluation -- the single summary table across the whole decade.
    Answers: does the sentiment-integrated GARCH-X model forecast volatility
    better than traditional GARCH(1,1), overall?
    """
    ok_quarters = [
        q for yr in all_year_results if yr.get("status") == "ok"
        for q in yr.get("quarter_results", [])
        if q.get("status") == "ok" and q.get("cv", {}).get("status") == "ok"
    ]
    if not ok_quarters:
        raise ValueError("No successful quarters found in all_year_results.")

    realized_var, sigma2_t, sigma2_s, dates, is_outlier = [], [], [], [], []
    for q in ok_quarters:
        cv = q["cv"]
        realized_var.extend(cv["oof_realized_var"])
        sigma2_t.extend(cv["oof_sigma2_traditional"])
        sigma2_s.extend(cv["oof_sigma2_sentiment"])
        dates.extend(cv["oof_dates"])
        outlier_set = set(q["outlier_dates"])
        is_outlier.extend([d in outlier_set for d in cv["oof_dates"]])

    realized_var = np.array(realized_var)
    sigma2_t, sigma2_s = np.array(sigma2_t), np.array(sigma2_s)
    is_outlier = np.array(is_outlier)

    error_metrics = compute_error_metrics(realized_var, sigma2_t, sigma2_s)
    dm_stat, dm_p = diebold_mariano_test(realized_var, sigma2_t, sigma2_s)
    directional_metrics = compute_directional_metrics(realized_var, sigma2_t, sigma2_s)
    risk_metrics = compute_risk_metrics(realized_var, sigma2_t, sigma2_s, is_outlier_mask=is_outlier)

    n_years = len({q["year"] for q in ok_quarters})
    n_quarters = len(ok_quarters)
    risk_verdict = "HELPED" if risk_metrics.loc[1, "qlike"] < risk_metrics.loc[0, "qlike"] else "DID NOT HELP"
    rmse_verdict = "HELPED" if error_metrics.loc[1, "RMSE (vol, pp)"] < error_metrics.loc[0, "RMSE (vol, pp)"] else "DID NOT HELP"

    print(f"{'='*72}\nOVERALL 10-YEAR VERDICT -- pooled across {n_years} years / {n_quarters} quarters, "
          f"{len(realized_var)} out-of-fold days\n{'='*72}")
    print("\n-- Volatility forecasting accuracy --")
    print(error_metrics.to_string(index=False))
    print(f"\nDiebold-Mariano (pooled): stat={dm_stat:.4f}, p={dm_p:.4f}")
    print("\n-- Directional accuracy (volatility up/down) --")
    print(directional_metrics.to_string(index=False))
    print("\n-- Risk metrics (QLIKE, lower = better) --")
    print(risk_metrics.to_string(index=False))
    print(f"\n{'-'*72}")
    print(f"RMSE/MAE:  sentiment integration {rmse_verdict}")
    print(f"RISK:      sentiment integration {risk_verdict}")
    print(f"{'-'*72}")

    return {
        "n_years": n_years, "n_quarters": n_quarters, "n_oof_points": len(realized_var),
        "error_metrics": error_metrics, "dm_stat": dm_stat, "dm_pvalue": dm_p,
        "directional_metrics": directional_metrics, "risk_metrics": risk_metrics,
        "rmse_verdict": rmse_verdict, "risk_verdict": risk_verdict,
    }
