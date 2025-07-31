import numpy as np
import pandas as pd
import os
import re
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion, market_implied_prior_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import risk_matrix

# ===============================================================
# Step 1: Load & Preprocess Historical Prices
# ===============================================================
"""
Load CRSP historical prices, ensure correct tickers (FB -> META),
and convert dates to datetime format for proper time series handling.
"""
df = pd.read_csv('data/Historical Price/CRSP_Historical_Price.csv')
df['TICKER'] = df['TICKER'].replace('FB', 'META')   
df['date'] = pd.to_datetime(df['date'])             

# ===============================================================
# Step 2: Build Adjusted Prices (Price Series with Cumulative Returns)
# ===============================================================
"""
Construct adjusted prices for each stock using cumulative returns.
This ensures price series are continuous even if raw prices/returns are missing.
"""
def build_adjusted_prices(group):
    group = group.copy().sort_values('date')
    group['RET'] = pd.to_numeric(group['RET'], errors='coerce')
    first_valid_price = group.loc[group['PRC'].first_valid_index(), 'PRC']
    rets = group['RET'].fillna(0)
    rets.iloc[0] = 0  # First day = initial price (no return)
    group['adj_price'] = first_valid_price * (1 + rets).cumprod()
    return group

df = df.groupby("TICKER", group_keys=False).apply(build_adjusted_prices)
market_prices = df.pivot(index='date', columns='TICKER', values='adj_price')

# ===============================================================
# Step 3: Build Covariance Matrix
# ===============================================================
"""
Use sample covariance of adjusted prices to construct the risk model.
This will be used for Black-Litterman and portfolio optimization.
"""
cov_matrix = risk_matrix(market_prices, method='sample_cov')

# ===============================================================
# Step 4: Define Market Caps & Weights
# ===============================================================
"""
Manually define the market caps for key tickers and convert them to weights.
These represent the 'market portfolio' baseline for Black-Litterman.
"""
mcaps = {
    "AAPL": 3.139e12, "AMD": 259.554e9, "AMZN": 2.369e12, "ASML": 323.879e9,
    "CRM": 246.6e9, "GOOGL": 2.225e12, "META": 1.767e12, "MSFT": 3.758e12,
    "NVDA": 4.179e12, "PANW": 128.419e9, "PLTR": 356.134e9, "SHOP": 155.212e9,
    "SNOW": 70.715e9, "TSLA": 1.036e12
}
total_mcap = sum(mcaps.values())
market_weights = {k: v / total_mcap for k, v in mcaps.items()}
tickers = list(mcaps.keys())

# ===============================================================
# Step 5: Compute Market-Implied Returns
# ===============================================================
"""
Estimate risk aversion (delta) and market-implied returns (prior),
which are inputs for the Black-Litterman model.
"""
delta = market_implied_risk_aversion(market_prices)
prior = market_implied_prior_returns(mcaps, delta, cov_matrix)

# ===============================================================
# Step 6: Load Investor Views & Construct P Matrices
# ===============================================================
"""
Load investor views (best_idea.csv). Build long-only P matrices:
 - Rows: Funds (investor views)
 - Columns: Tickers
 - Values: Normalized portfolio weights for each view
Also record 'confidence' for each view (e.g., CAPM market tilt).
"""
P_df = pd.read_csv("/Users/leon/Documents/GitHub/Black_Litterman_Dissertation/best_idea.csv")
P_df['CAPM_market_tilt'] = pd.to_numeric(P_df['CAPM_market_tilt'], errors='coerce')

# Helper: clean bracketed numeric values like "[0.05]"
def clean_brackets(x):
    if pd.isna(x): return None
    return float(re.sub(r'[\[\]]', '', str(x)))
P_df['future_pred'] = P_df['future_pred'].apply(clean_brackets)

quarters = P_df['future_quarter'].unique()
save_dir = "/Users/leon/Documents/GitHub/Black_Litterman_Dissertation/P_matrices_longonly"
os.makedirs(save_dir, exist_ok=True)

P_matrices, confidences = {}, {}
for q in quarters:
    df_q = P_df[P_df['future_quarter'] == q]
    tickers_q = df_q['ticker'].unique()
    funds = df_q['fund'].unique()
    P = pd.DataFrame(0.0, index=funds, columns=tickers_q)
    conf = pd.Series(index=funds, dtype=float)

    for fund in funds:
        df_fund = df_q[df_q['fund'] == fund]
        preds = df_fund.set_index('ticker')['future_pred'].dropna()
        if preds.empty: continue
        weights = preds / preds.sum()  # Long-only normalized weights
        P.loc[fund, weights.index] = weights
        conf[fund] = df_fund['CAPM_market_tilt'].mean()

    P_matrices[q] = P
    confidences[q] = conf
    P.to_csv(os.path.join(save_dir, f"P_matrix_{q}.csv"))
    conf.to_csv(os.path.join(save_dir, f"confidence_{q}.csv"))
print(f"Generated {len(P_matrices)} quarterly P matrices (long-only).")

# Pick one quarter for demonstration
P_matrix = P_matrices[quarters[0]]

# ===============================================================
# Step 7: Compute Historical Manager Returns (for Q)
# ===============================================================
"""
For each (fund, ticker), compute historical average quarterly returns
from 13F filings until the previous quarter. This forms the basis for Q.
"""
fund_data_path_template = "Data/13F/{ticker}_only_grouped.csv"
ticker_file_map = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "meta": "META",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA", "advanced_micro": "AMD",
    "palantir": "PLTR", "palo_alto": "PANW", "shopify": "SHOP",
    "snowflake": "SNOW", "salesforce": "CRM", "asml": "ASML"
}

def compute_manager_return(fund, file_ticker, prev_quarter, market_prices):
    file_path = fund_data_path_template.format(ticker=file_ticker)
    if not os.path.exists(file_path): return np.nan
    if file_ticker not in ticker_file_map: return np.nan
    market_ticker = ticker_file_map[file_ticker]

    df = pd.read_csv(file_path)
    df = df[df['FILINGMANAGER_NAME'] == fund]
    if df.empty: return np.nan
    df['REPORTCALENDARORQUARTER'] = pd.to_datetime(df['REPORTCALENDARORQUARTER'], format='%d-%b-%Y')
    df['QuarterLabel'] = df['REPORTCALENDARORQUARTER'].apply(lambda d: f"{d.year}(Q{(d.month-1)//3+1})")
    df = df[df['QuarterLabel'] < prev_quarter]
    if df.empty: return np.nan

    earliest_date = df['REPORTCALENDARORQUARTER'].min()
    if market_ticker not in market_prices.columns: return np.nan
    prices = market_prices[market_ticker].resample('QE').last()
    prices = prices[prices.index >= earliest_date]
    prices = prices[prices.index < pd.to_datetime(prev_quarter[:4]) + pd.offsets.QuarterEnd(int(prev_quarter[-2]))]
    returns = prices.pct_change().dropna()
    if returns.empty: return np.nan
    return returns.mean()

P_df['historical_avg_return'] = P_df.apply(
    lambda row: compute_manager_return(row['fund'], row['ticker'], row['prev_quarter'], market_prices),
    axis=1
)

# ===============================================================
# Step 8: Build Q Vectors and Embed into P Matrices
# ===============================================================
"""
For each quarter, compute Q (expected portfolio return per fund view) 
as the mean of historical average returns across that fund's tickers.
Append Q as a new column to the P matrix.
"""
Q_vectors = {}
for q in quarters:
    df_q = P_df[P_df['future_quarter'] == q]
    Q = df_q.groupby('fund')['historical_avg_return'].mean()  # could also blend with future_pred
    Q_vectors[q] = Q
    P_with_Q = P_matrices[q].copy()
    P_with_Q['Q'] = Q
    P_matrices[q] = P_with_Q
    P_with_Q.to_csv(os.path.join(save_dir, f"P_matrix_with_Q_{q}.csv"))
print(f"Updated {len(P_matrices)} P matrices with embedded Q.")

# For Black-Litterman: extract P (views matrix) and Q (vector)
P_matrix = P_matrices[quarters[0]].drop(columns=['Q'])
Q = P_matrices[quarters[0]]['Q'].values

# ===============================================================
# Step 9: Build Black-Litterman Model
# ===============================================================
"""
Combine market equilibrium (pi) and investor views (P, Q) 
to compute posterior expected returns using Black-Litterman.
"""
bl = BlackLittermanModel(
    cov_matrix=cov_matrix,
    pi="market",
    market_caps=market_weights,
    Q=Q,
    P=P_matrix
)
bl_return = bl.bl_returns()

# ===============================================================
# Step 10: Optimize Portfolio (Max Sharpe)
# ===============================================================
"""
Use the Black-Litterman posterior returns to build an efficient frontier 
and select the Max Sharpe portfolio.
"""
ef = EfficientFrontier(bl_return, cov_matrix)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# ===============================================================
# Step 11: Output Results
# ===============================================================
print("Optimized Portfolio Weights:")
print(cleaned_weights)
