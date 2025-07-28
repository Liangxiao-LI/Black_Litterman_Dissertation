import pandas as pd
import glob
import os

# 1. 获取所有 *_HP.csv 文件
csv_files = glob.glob("*_HP.csv")

# 2. 设置季度末区间
quarter_ends = pd.date_range(start="2020-03-31", end="2024-12-31", freq="Q")
quarter_labels = quarter_ends.to_period("Q").astype(str)  # '2020Q1', '2020Q2', ...

# 3. 初始化 DataFrame
quarterly_avg_prices = pd.DataFrame()

# 4. 处理每个文件
for file in csv_files:
    ticker = os.path.basename(file).split("_")[0]
    df = pd.read_csv(file)

    # 日期处理
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # 查找收盘价列
    price_col = None
    for col in ["Close", "Close/Last", "Adj Close", "close", "close/last"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        print(f"[跳过] {ticker}: 未找到收盘价列")
        continue

    # 按季度分组取均值
    df["Quarter"] = df.index.to_period("Q").astype(str)  # '2020Q1' 这种格式
    df[price_col] = df[price_col].replace('[\$,]', '', regex=True).astype(float)
    avg_prices = df.groupby("Quarter")[price_col].mean()

    # 添加到结果表
    quarterly_avg_prices.loc[ticker, avg_prices.index] = avg_prices.values

# 5. 保存季度平均价格
# 只保留在数据中实际出现过的季度列
available_quarters = [q for q in quarter_labels if q in quarterly_avg_prices.columns]
quarterly_avg_prices = quarterly_avg_prices[available_quarters]
quarterly_avg_prices.to_csv("quarterly_avg_prices.csv")
print("✅ 已生成 quarterly_avg_prices.csv")

# 6. 计算季度平均收益率
df = quarterly_avg_prices.copy()

# 清理符号，转为数值
df = df.replace('[\$,]', '', regex=True).apply(pd.to_numeric, errors='coerce')

# 环比百分比变化
pct_change_df = df.pct_change(axis=1) * 100
pct_change_df = pct_change_df.round(2)

# 去掉全为空的列（即某季度所有股票都没有数据）
pct_change_df = pct_change_df.dropna(axis=1, how='all')
# 保存
pct_change_df.to_csv("Real_pct_change_quarterly.csv")
print("✅ 已生成 Real_pct_change_quarterly.csv（基于季度平均价格）")