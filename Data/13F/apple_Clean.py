import pandas as pd

# 读取文件
df = pd.read_csv("apple_holdings_with_date.csv")

# 打印前5行
print(df.head())

# 删除 DATEREPORTED 列
df = df.drop(columns=["TITLEOFCLASS"])

# 打印前5行
print(df.head())

# 查看 NAMEOFISSUER 列的所有唯一值
print(df["NAMEOFISSUER"].unique())

# 只保留 NAMEOFISSUER 为 'APPLE INC' 或 'Apple Inc' 的行
apple_only_df = df[df["NAMEOFISSUER"].isin(["APPLE INC", "Apple Inc"])]
print(apple_only_df)