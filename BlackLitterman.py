import numpy as np
import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion, market_implied_prior_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import risk_matrix
from pypfopt.expected_returns import mean_historical_return

# ===========================
# Step 1: Load & Preprocess Historical Prices
# ===========================
df = pd.read_csv('data/Historical Price/CRSP_Historical_Price.csv')
df['TICKER'] = df['TICKER'].replace('FB', 'META')  # Replace old FB ticker with META
df['date'] = pd.to_datetime(df['date'])            # Ensure date column is datetime

# ===========================
# Step 2: Build Adjusted Prices (price adjusted by returns)
# ===========================
def build_adjusted_prices(group):
    group = group.copy().sort_values('date')
    group['RET'] = pd.to_numeric(group['RET'], errors='coerce')
    
    # Use the first valid price as the starting adjusted price
    first_valid_price = group.loc[group['PRC'].first_valid_index(), 'PRC']
    rets = group['RET'].fillna(0).copy()
    rets.iloc[0] = 0  # The first day price stays as the starting price
    
    # Cumulative product of returns to build adjusted price series
    group['adj_price'] = first_valid_price * (1 + rets).cumprod()
    return group

df = df.groupby("TICKER", group_keys=False).apply(build_adjusted_prices)

# Pivot to wide format: index = date, columns = tickers, values = adjusted prices
market_prices = df.pivot(index='date', columns='TICKER', values='adj_price')

# ===========================
# Step 3: Build Covariance Matrix
# ===========================
cov_matrix = risk_matrix(market_prices, method='sample_cov')  # Sample covariance

# ===========================
# Step 4: Define Market Caps & Weights
# ===========================
mcaps = {
    "AAPL": 3.139e12, "AMD": 259.554e9, "AMZN": 2.369e12, "ASML": 323.879e9,
    "CRM": 246.6e9, "GOOGL": 2.225e12, "META": 1.767e12, "MSFT": 3.758e12,
    "NVDA": 4.179e12, "PANW": 128.419e9, "PLTR": 356.134e9, "SHOP": 155.212e9,
    "SNOW": 70.715e9, "TSLA": 1.036e12
}
total_mcap = sum(mcaps.values())
market_weights = {k: v / total_mcap for k, v in mcaps.items()}  # Normalize market caps
tickers = list(mcaps.keys())  # Ensure consistent order

# ===========================
# Step 5: Market-Implied Parameters
# ===========================
delta = market_implied_risk_aversion(market_prices)            # Market risk aversion
prior = market_implied_prior_returns(mcaps, delta, cov_matrix) # Implied prior returns

# ===========================
# Step 6: Define Investor Views (Q & P)
# ===========================
# Relative views:
# View 1: META expected to outperform AMZN by 5%
# View 2: TSLA expected to outperform AAPL by 2%
Q = np.array([0.05, 0.02]).reshape(-1, 1)

import os
# 读取数据
import pandas as pd
import os

# 读取数据
P_df = pd.read_csv("/Users/leon/Documents/GitHub/Black_Litterman_Dissertation/best_idea.csv")

# 转换 future_pred 和 CAPM_market_tilt 为数值

P_df['CAPM_market_tilt'] = pd.to_numeric(P_df['CAPM_market_tilt'], errors='coerce')

# 获取所有 future_quarter
quarters = P_df['future_quarter'].unique()

# 保存目录
save_dir = "/Users/leon/Documents/GitHub/Black_Litterman_Dissertation/P_matrices_longonly"
os.makedirs(save_dir, exist_ok=True)

import re

def clean_brackets(x):
    if pd.isna(x):
        return None
    # 去掉中括号再转 float
    return float(re.sub(r'[\[\]]', '', str(x)))
P_df['future_pred'] = P_df['future_pred'].apply(clean_brackets)


P_matrices = {}
confidences = {}

for q in quarters:
    df_q = P_df[P_df['future_quarter'] == q]
    tickers = df_q['ticker'].unique()
    funds = df_q['fund'].unique()
    
    # 初始化 P 矩阵
    P = pd.DataFrame(0.0, index=funds, columns=tickers)
    conf = pd.Series(index=funds, dtype=float)
    


    for fund in funds:
        df_fund = df_q[df_q['fund'] == fund]
        preds = df_fund.set_index('ticker')['future_pred']
        
        # 去除空值
        preds = preds.dropna()
        if preds.empty:  # 如果没有数值跳过
            continue
        
        # 归一化（long-only，和为1）
        weights = preds / preds.sum()
        
        # 填入矩阵
        for ticker, val in weights.items():
            P.loc[fund, ticker] = val
        
        # confidence
        conf[fund] = df_fund['CAPM_market_tilt'].mean()
    
    P_matrices[q] = P
    confidences[q] = conf
    
    # 保存
    P.to_csv(os.path.join(save_dir, f"P_matrix_{q}.csv"))
    conf.to_csv(os.path.join(save_dir, f"confidence_{q}.csv"))

print(f"已生成 {len(P_matrices)} 个季度的 P 矩阵（long-only），存储在: {save_dir}")


# ===========================
# Step 7: Build Black-Litterman Model
# ===========================
bl = BlackLittermanModel(
    cov_matrix=cov_matrix,
    pi="market",
    market_caps=market_weights,
    Q=Q,
    P=P_matrix
)
bl_return = bl.bl_returns()  # Posterior returns

# ===========================
# Step 8: Optimize Portfolio (Max Sharpe)
# ===========================
ef = EfficientFrontier(bl_return, cov_matrix)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# ===========================
# Step 9: Output
# ===========================
print(cleaned_weights)