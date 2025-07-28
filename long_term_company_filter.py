import pandas as pd

# Replace with your actual file path
# 相对路径（从脚本所在目录出发）
df = pd.read_csv("Data/13F/apple_only_grouped.csv")


# Sort by manager and quarter
df['REPORTCALENDARORQUARTER'] = pd.to_datetime(df['REPORTCALENDARORQUARTER'])

# 筛选 2020 年之后的数据
df = df[df['REPORTCALENDARORQUARTER'] >= '2020-01-01']
df = df.sort_values(by=['FILINGMANAGER_NAME', 'REPORTCALENDARORQUARTER'])

# Calculate quarter-over-quarter change
df['QoQ_change'] = df.groupby('FILINGMANAGER_NAME')['VALUE'].pct_change().abs()

# Compute turnover proxy: average absolute change
turnover = df.groupby('FILINGMANAGER_NAME')['QoQ_change'].mean().reset_index()
turnover.columns = ['FILINGMANAGER_NAME', 'avg_turnover']

# 计算每个基金的季度数（持仓报告数量）
quarters_count = df.groupby('FILINGMANAGER_NAME')['REPORTCALENDARORQUARTER'].nunique().reset_index()
quarters_count.columns = ['FILINGMANAGER_NAME', 'quarters_held']

# 合并结果
fund_stats = pd.merge(turnover, quarters_count, on='FILINGMANAGER_NAME')


# Flag long-term managers (low turnover)
# 筛选长期基金：低换手 + 足够季度覆盖
long_term = fund_stats[(fund_stats['avg_turnover'] < 0.3) & (fund_stats['quarters_held'] > 10)]

print(long_term)

num_long_term = long_term['FILINGMANAGER_NAME'].nunique()
print(f"长期基金数量: {num_long_term}")

print(fund_stats.sort_values('avg_turnover'))  # 查看全部基金的换手率和季度数
print(fund_stats.sort_values('quarters_held'))  # 查看全部基金的换手率和季度数
print("\nLong-term funds:")
print(long_term)

####################################3

import pandas as pd
import glob
import os

# 参数
path = "Data/13F/"
turnover_threshold = 0.3
quarters_threshold = 8

# 找到所有 _only_grouped.csv 文件
files = glob.glob(os.path.join(path, "*_only_grouped.csv"))

all_results = []

for idx, file in enumerate(files, start=1):
    ticker = os.path.basename(file).replace("_only_grouped.csv", "")
    df = pd.read_csv(file)
    
    # 日期处理
    df['REPORTCALENDARORQUARTER'] = pd.to_datetime(
        df['REPORTCALENDARORQUARTER'], 
        errors='coerce'
    )
    df = df[df['REPORTCALENDARORQUARTER'] >= '2020-01-01']
    df = df.sort_values(by=['FILINGMANAGER_NAME', 'REPORTCALENDARORQUARTER'])
    
    # 换手率
    df['QoQ_change'] = df.groupby('FILINGMANAGER_NAME')['VALUE'].pct_change().abs()
    turnover = df.groupby('FILINGMANAGER_NAME')['QoQ_change'].mean().reset_index()
    turnover.columns = ['FILINGMANAGER_NAME', 'avg_turnover']
    
    # 季度数
    quarters_count = df.groupby('FILINGMANAGER_NAME')['REPORTCALENDARORQUARTER'].nunique().reset_index()
    quarters_count.columns = ['FILINGMANAGER_NAME', 'quarters_held']
    
    # 平均投资量
    avg_investment = df.groupby('FILINGMANAGER_NAME')['VALUE'].mean().reset_index()
    avg_investment.columns = ['FILINGMANAGER_NAME', 'avg_investment']
    
    # 合并
    fund_stats = turnover.merge(quarters_count, on='FILINGMANAGER_NAME')
    fund_stats = fund_stats.merge(avg_investment, on='FILINGMANAGER_NAME')
    
    # 筛选
    long_term = fund_stats[
        (fund_stats['avg_turnover'] < turnover_threshold) &
        (fund_stats['quarters_held'] > quarters_threshold)
    ].copy()
    long_term['ticker'] = ticker
    long_term['ticker_id'] = idx
    
    all_results.append(long_term)

# 合并所有结果
final_df = pd.concat(all_results, ignore_index=True)

# 统计基金出现在多少个 ticker
manager_counts = final_df.groupby('FILINGMANAGER_NAME')['ticker'].nunique().reset_index()
manager_counts.columns = ['FILINGMANAGER_NAME', 'tickers_invested']
final_df = final_df.merge(manager_counts, on='FILINGMANAGER_NAME', how='left')

final_df = final_df.sort_values(by=['FILINGMANAGER_NAME', 'ticker']).reset_index(drop=True)

# 输出
print(final_df.head())
print("\n每个基金出现在多少个 ticker 中：")
print(manager_counts.sort_values('tickers_invested', ascending=False))

final_df.to_csv("Data/13F/long_term_fund_stats_all_tickers.csv", index=False)

P_raw = final_df.pivot_table(
    index='FILINGMANAGER_NAME', 
    columns='ticker', 
    values='avg_investment', 
    aggfunc='sum',
    fill_value=0
)

# Step 2: Normalize each row so that weights sum to 1
P = P_raw.div(P_raw.sum(axis=1), axis=0)

# Step 3: Convert to numpy (optional, for Black-Litterman use)
P_matrix = P.values


# Step 6: 保存为 CSV
P.to_csv("P_matrix.csv")