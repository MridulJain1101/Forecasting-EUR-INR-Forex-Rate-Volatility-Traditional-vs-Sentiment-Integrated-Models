# Forecasting EUR/INR Forex Rate Volatility: Traditional vs Sentiment-Integrated Models

## About the Project

This project investigates whether financial news sentiment can improve
the forecasting of EUR/INR exchange rate volatility. It compares a
traditional **GARCH(1,1)** model, which relies solely on historical
exchange-rate data, with a sentiment-integrated **GARCH-X** model that
incorporates lagged ECB- and RBI-related news sentiment.

The analysis covers **2016--2025** and combines financial econometrics,
news analytics and statistical model evaluation. News sentiment is
derived from GDELT's article-level V2Tone measure, while financial
keyword analysis is conducted using the Loughran--McDonald financial
dictionary.

The objective is not only to compare forecast errors, but also to
examine whether news sentiment helps identify the direction of
volatility changes and which financial news themes are associated with
volatility shocks.

## About the Dataset

The project combines exchange-rate history, news sentiment data and a
financial vocabulary resource.

### 1. EUR/INR Exchange Rate Data

-   **Coverage:** 1 January 2015 -- 1 January 2026.
-   **Frequency:** Daily.
-   **Purpose:** Used to calculate daily log returns and the
    realised-volatility proxy.
-   The 2015 observations provide a year of historical volatility
    information before the 2016 forecasting period begins.
-   The thesis describes the exchange-rate series as being retrieved
    through a dedicated data-retrieval function in the project pipeline.
    The exchange-rate provider should be documented here once confirmed
    from the source code or original data files.

### 2. ECB and RBI News Sentiment Data

-   **Coverage:** 2016--2025.
-   **Source:** [GDELT Project](https://www.gdeltproject.org/).
-   **Measure:** GDELT's article-level `V2Tone` field.
-   Two separate news streams are constructed:
    -   **ECB stream:** News related to ECB communication and euro-area
        monetary and macroeconomic conditions.
    -   **RBI stream:** News related to RBI policy and Indian
        rupee/macroeconomic conditions.
-   Fixed keyword and regular-expression filters are applied to identify
    relevant articles. Daily sentiment is calculated from the tone
    scores of matching articles.
-   The daily ECB and RBI sentiment variables are lagged by one trading
    day before entering the forecasting model to reduce look-ahead bias.

### 3. Financial Keyword Dictionary

-   **Resource:** Loughran--McDonald financial dictionary
    (`Loughran-McDonald_MasterDictionary_1993-2025`).
-   **Purpose:** Restricts the vocabulary used in the TF-IDF
    keyword-attribution analysis to finance-relevant terms.
-   **Important:** The dictionary is **not** used to calculate the GDELT
    sentiment scores. Sentiment comes directly from GDELT's `V2Tone`
    field.

### 4. Google BigQuery News Retrieval

Google BigQuery is used to retrieve underlying news quotations for
qualitative review of statistically identified volatility-shock dates.
Full quotation retrieval is performed for flagged shock days, while a
sample of normal days is retrieved for comparison. BigQuery is not used
to calculate the daily GDELT sentiment averages.

> **Data availability:** Add the original data files or instructions for
> obtaining them if you are permitted to redistribute them. Do not
> commit Google Cloud service-account credentials, API keys or other
> secrets to this repository.

## Analysis Approach & Methodology

The analysis follows a structured pipeline, from data preparation and
volatility measurement to econometric modelling, out-of-sample
evaluation and news attribution.

### 1. Data Preparation and Integration

-   Combine daily EUR/INR exchange-rate observations with the ECB and
    RBI news-sentiment series by calendar date.
-   Calculate daily log returns from the exchange-rate levels.
-   Construct the realised-volatility proxy using the square root of
    squared demeaned daily returns.
-   Lag both sentiment series by one period before using them as model
    inputs.
-   Retain 2015 exchange-rate observations as the historical baseline;
    the sentiment-forecasting evaluation covers 2016--2025.

### 2. Volatility-Shock Identification

A date is classified as a volatility-shock day when its realised
volatility exceeds an expanding historical baseline by more than **two
standard deviations**. The expanding baseline uses information available
up to the date being assessed.

The resulting shock labels are used in two ways: - To compare model
behaviour on shock days and normal days. - To separate shock-day and
normal-day news for the TF-IDF keyword-attribution analysis.

### 3. Traditional Benchmark: GARCH(1,1)

A traditional GARCH(1,1) model is estimated using EUR/INR return history
only. It provides the benchmark against which the sentiment-integrated
model is evaluated.

The conditional variance is specified as:

``` text
σ²(t) = ω + α ε²(t−1) + β σ²(t−1)
```

### 4. Sentiment-Integrated Model: GARCH-X

The GARCH-X model extends the traditional variance equation by
incorporating the one-period-lagged ECB and RBI sentiment scores as
separate exogenous variables:

``` text
σ²(t) = ω + α ε²(t−1) + β σ²(t−1)
        + γ(ECB) × Sentiment(ECB,t−1)
        + γ(RBI) × Sentiment(RBI,t−1)
```

The model makes it possible to examine whether ECB- and RBI-related news
sentiment contains information beyond historical exchange-rate
movements. The estimated sentiment coefficients and their p-values are
also analysed.

### 5. Walk-Forward Forecasting and Model Fitting

The comparison uses a **quarter-by-quarter walk-forward procedure**
across the ten-year period: - Divide 2016--2025 into 40 quarterly
evaluation windows. - Generate expanding-window, one-step-ahead
out-of-sample forecasts within each quarter. - Fit separate in-sample
quarterly models to report model coefficients. - Fit full-year in-sample
models to obtain annual coefficients and generate the yearly
fitted-volatility plots.

The in-sample coefficient fits and yearly plots are kept separate from
the out-of-sample forecasting evaluation.

### 6. Statistical Diagnostics and Forecast Evaluation

Before model estimation, the Ljung--Box test is applied to squared,
demeaned daily returns (10 lags) to assess whether volatility clustering
is present.

The models are compared using:

  -----------------------------------------------------------------------
  Metric / Test                       Purpose
  ----------------------------------- -----------------------------------
  RMSE                                Measures forecast error, with a
                                      greater penalty for large errors

  MAE                                 Measures average absolute forecast
                                      error

  MAPE                                Measures percentage forecast error;
                                      treated as a supporting metric
                                      because it can become unstable when
                                      realised volatility is close to
                                      zero

  QLIKE                               Evaluates volatility forecasts
                                      using a loss function designed for
                                      volatility measurement

  Diebold--Mariano test               Tests whether the difference in
                                      forecast accuracy between the two
                                      models is statistically significant

  Directional accuracy                Measures how often the model
                                      correctly predicts whether
                                      volatility rises or falls

  Directional F1 score                Provides an additional measure of
                                      directional classification
                                      performance

  Ljung--Box test                     Checks for autocorrelation in
                                      squared returns as a diagnostic for
                                      volatility clustering
  -----------------------------------------------------------------------

Forecast errors from all 40 quarterly windows are pooled for the overall
Diebold--Mariano comparison.

### 7. TF-IDF Keyword Attribution and Event Review

-   Split news coverage into volatility-shock and normal-day groups.
-   Apply TF-IDF to identify financial terms that distinguish the two
    groups.
-   Restrict the vocabulary to the Loughran--McDonald financial
    dictionary.
-   Compare keyword weights and document frequency to identify recurring
    themes.
-   Use Google BigQuery to retrieve relevant news quotations for manual
    review of selected shock dates and their contemporaneous events.

The event review is a qualitative, keyword-guided assessment. It does
not establish that a particular news article caused a volatility shock.

## Conclusion

The pooled out-of-sample results show that adding GDELT-derived
sentiment did **not** improve overall EUR/INR volatility-forecasting
accuracy over the traditional GARCH(1,1) benchmark.

Across 800 out-of-fold observations from 40 quarterly windows:

  Evaluation Metric        Traditional GARCH(1,1)   Sentiment GARCH-X
  ---------------------- ------------------------ -------------------
  RMSE                                 **0.4310**              0.4577
  MAE                                  **0.3342**              0.3576
  MAPE                                 **2234.5**              2266.0
  QLIKE                               **-0.0853**              0.8640
  Directional accuracy                     0.3942          **0.4668**
  Directional F1 score                     0.3920          **0.4668**

The pooled Diebold--Mariano statistic is **-2.7400 (p = 0.0061)**,
favouring the traditional GARCH(1,1) model on forecast loss. The
sentiment-integrated model performs better on directional accuracy and
is somewhat more responsive on shock days, but this does not translate
into lower overall forecast error.

The findings suggest that news sentiment may provide some directional
information, but its forecasting value is not consistent across the full
sample. Its coefficients also change sign across estimation windows,
limiting the stability of the relationship.

### Limitations and Future Scope

-   GDELT's pre-computed V2Tone scores are used directly and are not
    independently validated against a separately labelled financial-news
    dataset.
-   Fixed keyword filters may exclude relevant articles or include
    articles that match the terms in an unrelated context.
-   BigQuery cost constraints mean full news-text retrieval is
    concentrated on shock days, with only a sample of normal days
    retrieved.
-   GDELT's source coverage may be uneven, particularly between European
    and Indian news.
-   The current GARCH-X specification is symmetric and linear.

Potential extensions include testing domain-specific sentiment models
such as FinBERT, using more complete news-text coverage, comparing
alternative news providers, and evaluating asymmetric or multiplicative
volatility models such as EGARCH, GJR-GARCH or NA-GARCH.

## Tools and Techniques

### Programming Language

-   **Python**

### Data Handling and Processing

-   `pandas`
-   `numpy`
-   `pickle`

### Econometric Modelling

-   `arch` -- GARCH(1,1) and GARCH-X estimation
-   Custom Python pipeline module (`fx_volatility_pipeline`)

### Data Extraction and Querying

-   `google-cloud-bigquery`
-   `google-auth`
-   Google BigQuery -- retrieval of news quotations for event review

### Statistical Analysis

-   `scipy.stats` -- statistical distributions and Ljung--Box test
    p-value calculations
-   GARCH-family maximum-likelihood estimation
-   TF-IDF keyword analysis
-   Walk-forward out-of-sample forecasting
-   Diebold--Mariano forecast comparison

### Evaluation Techniques

-   RMSE
-   MAE
-   MAPE
-   QLIKE
-   Directional accuracy
-   Directional F1 score
-   Ljung--Box diagnostic test
-   Statistical significance testing using p-values

### Data Visualisation

-   `matplotlib` -- yearly realised-volatility and fitted-model plots

------------------------------------------------------------------------

*This project is based on my Master's thesis, "Forecasting EUR/INR Forex
Rate Volatility: Traditional Model vs Sentiment Integrated Model"
(2026).*
