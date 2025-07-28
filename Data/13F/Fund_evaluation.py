import pandas as pd

# 1. 读取数据
df = pd.read_csv("apple_only_grouped.csv")

# 2. 分组计算
# 每个 hedge fund 持仓的季度数量
quarter_counts = df.groupby("FILINGMANAGER_NAME")["REPORTCALENDARORQUARTER"].nunique()

# 每个 hedge fund 平均每季度持有金额
avg_holdings = df.groupby("FILINGMANAGER_NAME")["VALUE"].mean()

# 每个 hedge fund 最大单季度持仓金额
max_holdings = df.groupby("FILINGMANAGER_NAME")["VALUE"].max()

# 3. 整合成一个 DataFrame
summary_df = pd.DataFrame({
    "Quarter_Count": quarter_counts,
    "Average_Holdings_Per_Quarter": avg_holdings,
    "Max_Holdings_Single_Quarter": max_holdings
})

# 4. 保存为 CSV
summary_df.to_csv("apple_hedge_fund_impact_summary.csv")



import pandas as pd
import os
import glob

# 1. Load real return data
real_return = pd.read_csv("/Users/leon/Documents/GitHub/Black_Litterman_Dissertation/Data/Historical Price/Real_pct_change.csv", index_col=0)

def convert_date_to_quarter(date_str):
    # date_str like '31/12/2020'
    day, month, year = map(int, date_str.split("/"))
    if month in [1, 2, 3]:
        quarter = 'Q1'
    elif month in [4, 5, 6]:
        quarter = 'Q2'
    elif month in [7, 8, 9]:
        quarter = 'Q3'
    else:
        quarter = 'Q4'
    return f"{year}Q{quarter[-1]}"

# Map company names to tickers
name_to_ticker = {
    "amazon": "AMZN",
    "snowflake": "SNOW",
    "apple": "AAPL",
    "tesla": "TSLA",
    "meta": "META",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "shopify": "SHOP",
    "palantir": "PLTR",
    "asml": "ASML",
    "salesforce": "CRM",
    "nvidia": "NVDA",
    "palo": "PANW",
    "advanced": "AMD"
}

# 2. Create a summary DataFrame
summary_scores = {}

# Iterate through all pct_change matrix files
for file_path in glob.glob("*_fund_pct_change_matrix.csv"):
    # Get company name and map to ticker
    company_name = file_path.split("_fund_pct_change_matrix.csv")[0].lower()
    ticker = name_to_ticker.get(company_name)
    if not ticker:
        print(f"Skipping: {company_name} (no ticker mapping found)")
        continue

    # Read fund matrix (rows = hedge funds, columns = dates)
    pct_matrix = pd.read_csv(file_path, index_col=0)

    # Real returns: Series of returns for that ticker
    if ticker not in real_return.index:
        print(f"Skipping: {ticker} not found in real_return")
        continue
    real_change = real_return.loc[ticker]

    # Convert dates to quarters
    real_change.index = real_change.index.map(convert_date_to_quarter)

    # Align on available dates
    aligned_dates = pct_matrix.columns.intersection(real_change.index)
    pct_matrix_aligned = pct_matrix[aligned_dates]
    real_change_aligned = real_change[aligned_dates]

    # 处理 inf、-inf 和 NaN：从无到有视为200%，从有到无视为-200%，其余 NaN 设为 0
    pct_matrix_aligned = pct_matrix_aligned.replace(float('inf'), 2.0)
    pct_matrix_aligned = pct_matrix_aligned.replace(float('-inf'), -2.0)
    pct_matrix_aligned = pct_matrix_aligned.fillna(0)

    # 单位统一（如果需要）：百分比转为小数
    pct_matrix_aligned = pct_matrix_aligned / 100
    real_change_aligned = real_change_aligned.astype(float) / 100

    # 计算得分
    scores = pct_matrix_aligned.mul(real_change_aligned.values, axis=1).sum(axis=1)
    summary_scores[ticker] = scores


    # Calculate scores
    scores = pct_matrix_aligned.mul(real_change_aligned.values, axis=1).sum(axis=1)
    summary_scores[ticker] = scores

# 4. Combine into one DataFrame
summary_df = pd.DataFrame(summary_scores).fillna(0)

# 5. Save to CSV
summary_df.to_csv("hedge_fund_impact_summary.csv")