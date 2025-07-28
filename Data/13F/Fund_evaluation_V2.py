import pandas as pd
import glob
import os

# === 读取真实季度收益 ===
real_return = pd.read_csv("/Users/leon/Documents/GitHub/Black_Litterman_Dissertation/Data/Historical Price/Real_pct_change.csv", index_col=0)

def convert_date_to_quarter(date_str):
    day, month, year = map(int, date_str.split("/"))
    if month in [1, 2, 3]: q = "Q1"
    elif month in [4, 5, 6]: q = "Q2"
    elif month in [7, 8, 9]: q = "Q3"
    else: q = "Q4"
    return f"{year}{q}"

# === Ticker映射 ===
name_to_ticker = {
    "amazon": "AMZN", "snowflake": "SNOW", "apple": "AAPL", "tesla": "TSLA",
    "meta": "META", "google": "GOOGL", "microsoft": "MSFT", "shopify": "SHOP",
    "palantir": "PLTR", "asml": "ASML", "salesforce": "CRM", "nvidia": "NVDA",
    "palo": "PANW", "advanced": "AMD","palo_alto": "PANW","advanced_micro": "AMD"  
}

# === 初始化总结果 ===
all_merged = pd.DataFrame()

# === 遍历每个公司（基于 *_only_grouped.csv 和 *_fund_pct_change_matrix.csv）===
grouped_files = glob.glob("*_only_grouped.csv")

for grouped_file in grouped_files:
    base_name = os.path.basename(grouped_file).replace("_only_grouped.csv", "").lower()
    ticker = name_to_ticker.get(base_name)
    if not ticker:
        print(f"⏭️ Skipping {base_name}: no ticker mapping")
        continue

    # === 1. 读取 grouped 持仓数据，计算三项统计 ===
    df = pd.read_csv(grouped_file)
    quarter_counts = df.groupby("FILINGMANAGER_NAME")["REPORTCALENDARORQUARTER"].nunique()
    avg_holdings = df.groupby("FILINGMANAGER_NAME")["VALUE"].mean()
    max_holdings = df.groupby("FILINGMANAGER_NAME")["VALUE"].max()

    stats_df = pd.DataFrame({
        f"Quarter_Count_{ticker}": quarter_counts,
        f"Average_Holdings_Per_Quarter_{ticker}": avg_holdings,
        f"Max_Holdings_Single_Quarter_{ticker}": max_holdings
    })

    # === 2. 读取持仓变化矩阵和得分 ===
    possible_matrix_files = glob.glob(f"{base_name.split('_')[0]}*_fund_pct_change_matrix.csv")
    if not possible_matrix_files:
        print(f"⚠️ Missing matrix file for base: {base_name}")
        continue
    matrix_file = possible_matrix_files[0]
    if not os.path.exists(matrix_file):
        print(f"⚠️ Missing matrix file: {matrix_file}")
        continue

    pct_matrix = pd.read_csv(matrix_file, index_col=0)

    if ticker not in real_return.index:
        print(f"⚠️ {ticker} not in real_return")
        continue
    real_change = real_return.loc[ticker]
    real_change.index = real_change.index.map(convert_date_to_quarter)

    aligned_dates = pct_matrix.columns.intersection(real_change.index)
    pct_matrix_aligned = pct_matrix[aligned_dates]
    real_change_aligned = real_change[aligned_dates]

    # 清洗并归一
    pct_matrix_aligned = pct_matrix_aligned.replace(float('inf'), 2.0)
    pct_matrix_aligned = pct_matrix_aligned.replace(float('-inf'), -2.0)
    pct_matrix_aligned = pct_matrix_aligned.fillna(0) / 100
    real_change_aligned = real_change_aligned.astype(float) / 100

    # 得分
    scores = pct_matrix_aligned.mul(real_change_aligned.values, axis=1).sum(axis=1)
    scores_df = pd.DataFrame({f"Score_{ticker}": scores})

    # === 合并所有列 ===
    company_df = stats_df.join(scores_df, how="outer").fillna(0)
    all_merged = all_merged.join(company_df, how="outer").fillna(0)  # 按 hedge fund 对齐

# === 保存最终结果 ===
all_merged.index.name = "FILINGMANAGER_NAME"
all_merged.to_csv("hedge_fund_impact_summary.csv")
print("✅ 完成：已保存 hedge_fund_impact_summary.csv")