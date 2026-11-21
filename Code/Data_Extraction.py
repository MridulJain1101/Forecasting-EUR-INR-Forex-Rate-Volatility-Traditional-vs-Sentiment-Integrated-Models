#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing the necessary libraries

# In[7]:


import time
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import numpy as np
import matplotlib.pyplot as plt
import requests
import calendar

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ### 2. Authentication for connection with Big Query

# In[2]:


key_path = "C:/Users/Mridul Jain/Master Thesis/Main/master-thesis-503909-16879b5b8359.json"

credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
)

print("Authenticated. Project:", client.project)


# ### 3. Setting up keyword lists for ECB and RBI

# In[3]:


ecb_keywords = [
    "ECB", "eurozone inflation", "deposit facility rate", "hawkish", "dovish",
    "quantitative tightening", "quantitative easing", "German inflation",
    "eurozone recession", "PMI eurozone", "energy crisis",
    "natural gas prices Europe", "Ukraine war sanctions", "euro area GDP",
    "ECB press conference"
]

rbi_keywords = [
    "RBI", "repo rate", "CPI inflation India", "rupee depreciation",
    "current account deficit", "FII outflows", "crude oil import bill",
    "monetary policy committee", "forex reserves RBI", "India GDP growth",
    "rupee intervention", "Fed rate hike spillover", "India current account",
    "RBI governor statement"
]

def build_keyword_regex(keywords):
    # Escape special regex characters and join with OR
    # \b adds word boundaries so "RBI" only matches as a standalone word,
    # not as a substring inside unrelated words like "Serbia" or "Barbie"
    escaped = [r"\b" + kw.replace(" ", r"\s+") + r"\b" for kw in keywords]
    return "(?i)(" + "|".join(escaped) + ")"

ecb_pattern = build_keyword_regex(ecb_keywords)
rbi_pattern = build_keyword_regex(rbi_keywords)


# ### 4. Query builder (uses partitioned table)

# In[4]:


def build_query(pattern, start_date, end_date):
    return f"""
    SELECT
      DATE,
      SourceCommonName,
      DocumentIdentifier AS SourceURL,
      V2Organizations,
      AllNames,
      V2Tone
    FROM
      `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
      _PARTITIONTIME >= TIMESTAMP("{start_date}")
      AND _PARTITIONTIME <= TIMESTAMP("{end_date}")
      AND REGEXP_CONTAINS(AllNames, r'{pattern}')
    """


# ### 5. Dry-run cost check

# In[5]:


def check_cost(query):
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    return job.total_bytes_processed / 1e9


# ### 6. Date ranges for 2016-2025 (full 10-year span)

# In[8]:


# ---------------------------------------------------------
# Generate monthly (start, end) date ranges for Jan 2016 - Dec 2025
# Starting from 2016 avoids the partial-month/data-gap complexity
# from Feb 2015 (that partial year was already extracted separately
# in an earlier notebook), giving a clean, complete 10-year span here
# ---------------------------------------------------------

def generate_monthly_periods(start_year, end_year):
    periods = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            start_str = f"{year}-{month:02d}-01"
            end_str = f"{year}-{month:02d}-{last_day:02d}"
            periods.append((start_str, end_str))
    return periods

all_periods_2016_2025 = generate_monthly_periods(2016, 2025)

print(f"Total months to extract: {len(all_periods_2016_2025)}")  # should print 120
print("First 3 periods:", all_periods_2016_2025[:3])
print("Last 3 periods:", all_periods_2016_2025[-3:])


# ### 7. Extraction loop (reusable function, saves one CSV per year)

# In[9]:


# ---------------------------------------------------------
# Run extraction year-by-year and save each year's data to its
# own CSV immediately after completion. If the process stops
# partway (quota limit, connection issue, etc.), you keep
# everything already extracted instead of losing it all.
#
# gb_limit: max GB allowed for a single month's query before it
#           gets skipped
# running_total_limit: safety stop for the WHOLE run, in GB, to
#           stay under the 1 TB/month free tier (set slightly
#           below 1000 as a safety margin)
# ---------------------------------------------------------

def extract_multi_year(pattern, keyword_label, periods, gb_limit=50, running_total_limit=950):
    grand_total_gb = 0.0
    current_year = None
    year_dfs = []

    for start, end in periods:
        year = start[:4]

        # When we cross into a new year, save the previous year's data first
        if current_year is not None and year != current_year:
            if year_dfs:
                year_combined = pd.concat(year_dfs, ignore_index=True)
                filename = f"{keyword_label.lower()}_gdelt_{current_year}.csv"
                year_combined.to_csv(filename, index=False)
                print(f"\n>>> Saved {filename} ({len(year_combined)} rows) <<<\n")
            year_dfs = []

        current_year = year

        q = build_query(pattern, start, end)
        gb = check_cost(q)
        grand_total_gb += gb
        print(f"[{keyword_label}] {start} to {end}: {gb:.2f} GB (running total: {grand_total_gb:.2f} GB)")

        if grand_total_gb > running_total_limit:
            print(f"\nSTOPPING: cumulative scan ({grand_total_gb:.2f} GB) is approaching the "
                  f"{running_total_limit} GB safety limit. Save progress and resume later "
                  f"(e.g., next month, once quota refreshes).")
            break

        if gb > gb_limit:
            print(f"  --> Skipped this month (exceeds {gb_limit} GB per-query limit)")
            continue

        df = client.query(q).to_dataframe()
        print(f"  --> Retrieved {len(df)} rows")
        year_dfs.append(df)
        time.sleep(1)

    # Save whatever's left for the final year in the loop
    if year_dfs:
        year_combined = pd.concat(year_dfs, ignore_index=True)
        filename = f"{keyword_label.lower()}_gdelt_{current_year}.csv"
        year_combined.to_csv(filename, index=False)
        print(f"\n>>> Saved {filename} ({len(year_combined)} rows) <<<\n")

    print(f"\n[{keyword_label}] EXTRACTION RUN COMPLETE")
    print(f"[{keyword_label}] Total GB scanned this run: {grand_total_gb:.2f} GB")

    return grand_total_gb


# ### 8. Test run — single year (2016 only) to estimate scale before committing

# In[10]:


# ---------------------------------------------------------
# Test on just 2016 first to see the actual GB-per-year rate.
# This tells us whether all 10 years x 2 streams will fit inside
# the 1 TB/month free quota, or whether the full run needs to be
# spread across multiple months.
# ---------------------------------------------------------

test_periods_2016 = generate_monthly_periods(2016, 2016)

test_gb = extract_multi_year(ecb_pattern, "ECB_TEST_2016", test_periods_2016)

print(f"\nEstimated GB for one year, one stream: {test_gb:.2f} GB")
print(f"Rough projection for 10 years, both streams: {test_gb * 10 * 2:.2f} GB")


# ### 9. Run full 2016-2025 extraction — ECB

# In[11]:


ecb_total_gb = extract_multi_year(ecb_pattern, "ECB", all_periods_2016_2025)


# In[13]:


# Your existing client from before stays exactly as it is -- no changes needed
# client = bigquery.Client(...)  <- this already exists in your notebook

# New second account/project
key_path_2 = "C:/Users/Mridul Jain/Master Thesis/Main/master-thesis-2-505017-bf07547ceaf5.json"

credentials_2 = service_account.Credentials.from_service_account_file(
    key_path_2,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client_2 = bigquery.Client(
    credentials=credentials_2,
    project=credentials_2.project_id,
)

print("Authenticated. Project:", client_2.project)


# In[18]:


def extract_single_year(pattern, keyword_label, periods, client, gb_limit=50):
    year_dfs = []
    total_gb = 0.0
    
    # Pull the actual year from the periods list instead of hardcoding it
    year = periods[0][0][:4]

    for start, end in periods:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        q = build_query(pattern, start, end)
        job = client.query(q, job_config=job_config)
        gb = job.total_bytes_processed / 1e9
        total_gb += gb
        print(f"[{keyword_label}] {start} to {end}: {gb:.2f} GB (running total: {total_gb:.2f} GB)")

        if gb > gb_limit:
            print(f"  --> Skipped (exceeds {gb_limit} GB limit)")
            continue

        df = client.query(q).to_dataframe()
        print(f"  --> Retrieved {len(df)} rows")
        year_dfs.append(df)
        time.sleep(1)

    if year_dfs:
        combined = pd.concat(year_dfs, ignore_index=True)
        filename = f"{keyword_label.lower()}_gdelt_{year}.csv"   # <-- now uses the real year
        combined.to_csv(filename, index=False)
        print(f"\n>>> Saved {filename} ({len(combined)} rows) <<<")

    print(f"\nTotal GB used for this extraction: {total_gb:.2f} GB")
    return total_gb


# In[19]:


# Re-extract 2018 (since it may have just been overwritten with 2019 data)
periods_2018_only = generate_monthly_periods(2018, 2018)
gb_used_2018 = extract_single_year(ecb_pattern, "ECB", periods_2018_only, client)


# In[20]:


# ---------------------------------------------------------
# Extract 2019 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2019_only = generate_monthly_periods(2019, 2019)

gb_used_2019 = extract_single_year(ecb_pattern, "ECB", periods_2019_only, client_2)


# In[22]:


# ---------------------------------------------------------
# Extract 2020 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2020_only = generate_monthly_periods(2020, 2020)

gb_used_2020 = extract_single_year(ecb_pattern, "ECB", periods_2020_only, client_2)


# In[23]:


# ---------------------------------------------------------
# Extract 2021 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2021_only = generate_monthly_periods(2021, 2021)

gb_used_2021 = extract_single_year(ecb_pattern, "ECB", periods_2021_only, client_2)


# In[24]:


# ---------------------------------------------------------
# Extract 2022 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2022_only = generate_monthly_periods(2022, 2022)

gb_used_2022 = extract_single_year(ecb_pattern, "ECB", periods_2022_only, client_2)


# In[25]:


# ---------------------------------------------------------
# Extract 2023 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2023_only = generate_monthly_periods(2023, 2023)

gb_used_2023 = extract_single_year(ecb_pattern, "ECB", periods_2023_only, client_2)


# In[26]:


# ---------------------------------------------------------
# Extract 2024 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2024_only = generate_monthly_periods(2024, 2024)

gb_used_2024 = extract_single_year(ecb_pattern, "ECB", periods_2024_only, client_2)


# In[27]:


# ---------------------------------------------------------
# Extract 2025 for ECB, using the new client (account 2)
# ---------------------------------------------------------

periods_2025_only = generate_monthly_periods(2025, 2025)

gb_used_2025 = extract_single_year(ecb_pattern, "ECB", periods_2025_only, client_2)


# In[29]:


# Now for RBI


# In[28]:


def run_rbi_year(year, client_to_use):
    periods = generate_monthly_periods(year, year)
    gb_used = extract_single_year(rbi_pattern, "RBI", periods, client_to_use)
    return gb_used

# Then just call, one line per year:
run_rbi_year(2016, client_2)


# In[30]:


run_rbi_year(2017, client_2)


# In[31]:


run_rbi_year(2018, client_2)


# In[32]:


run_rbi_year(2019, client_2)


# In[33]:


run_rbi_year(2020, client_2)


# In[34]:


run_rbi_year(2021, client_2)


# In[35]:


run_rbi_year(2022, client_2)


# In[37]:


# New third account/project
key_path_3 = "C:/Users/Mridul Jain/Master Thesis/Main/master-thesis-3-505019-5ccadc465d13.json"

credentials_3 = service_account.Credentials.from_service_account_file(
    key_path_3,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client_3 = bigquery.Client(
    credentials=credentials_3,
    project=credentials_3.project_id,
)

print("Authenticated. Project:", client_3.project)


# In[38]:


run_rbi_year(2023, client_3)


# In[39]:


run_rbi_year(2024, client_3)


# In[40]:


run_rbi_year(2025, client_3)


# ### Getting some information about the data

# In[1]:


# Load the CSVs
ecb_df = pd.read_csv("C:/Users/Mridul Jain/Master Thesis/Main/ECB_Data/ecb_gdelt_2016.csv")
rbi_df = pd.read_csv("C:/Users/Mridul Jain/Master Thesis/Main/RBI_Data/rbi_gdelt_2016.csv")

# Basic shape and structure
print("ECB data shape:", ecb_df.shape)
print("RBI data shape:", rbi_df.shape)

print("\nColumns:", ecb_df.columns.tolist())
print("\nData types:\n", ecb_df.dtypes)

# View first few rows
ecb_df.head(10)
rbi_df.head(10)


# In[ ]:




