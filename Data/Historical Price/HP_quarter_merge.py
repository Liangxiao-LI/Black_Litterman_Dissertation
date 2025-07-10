import pandas as pd
import glob
import os

# 1. 获取所有 *_HP.csv 文件（确保在同一目录）
csv_files = glob.glob("*_HP.csv")

# 2. 设置季度末列表：2020/03/31 到 2024/12/31，每季度末
quarter_ends = pd.date_range(start="2020-03-31", end="2024-12-31", freq="Q")
quarter_ends_str = quarter_ends.strftime("%Y-%m-%d")  # 转换为字符串用于列名

# 3. 初始化结果 DataFrame
all_prices = pd.DataFrame(index=[], columns=quarter_ends_str)

# 4. 批量处理每个股票文件
for file in csv_files:
    # 提取股票名（如 AAPL）
    ticker = os.path.basename(file).split("_")[0]

    # 读取 CSV 并处理日期
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # 检查实际使用的收盘价列名（兼容 'Close', 'Close/Last', 'Adj Close'）
    price_col = None
    for candidate in ["Close", "Close/Last", "Adj Close", "close", "close/last"]:
        if candidate in df.columns:
            price_col = candidate
            break

    if price_col is None:
        print(f"[跳过] {ticker}: 未找到收盘价列")
        continue

    # 提取季度末或前一交易日的价格
    prices = {}
    for date in quarter_ends:
        if date in df.index:
            price = df.loc[date, price_col]
        else:
            prev_date = df.index[df.index <= date].max()
            price = df.loc[prev_date, price_col] if pd.notna(prev_date) else None
        prices[date.strftime("%Y-%m-%d")] = price

    # 将当前股票加入总表
    all_prices.loc[ticker] = prices

# 5. 保存最终结果
all_prices.to_csv("quarterly_prices.csv")
print("✅ 已生成 quarterly_prices.csv")

# 1. 读取季度价格数据
df = pd.read_csv("quarterly_prices.csv", index_col=0)

# 去掉美元符号和千分位逗号
df = df.replace('[\$,]', '', regex=True)

# 2. 将所有值转换为 float（防止字符串报错）
df = df.apply(pd.to_numeric, errors="coerce")

# 2. 计算季度百分比变化（环比 %）
pct_change_df = df.pct_change(axis=1) * 100  # axis=1 表示对列进行变化率计算

# 3. 保留小数点两位（可选）
pct_change_df = pct_change_df.round(2)

# 4. 保存结果
pct_change_df.to_csv("Real_pct_change.csv")

print("✅ 已生成 Real_pct_change.csv")