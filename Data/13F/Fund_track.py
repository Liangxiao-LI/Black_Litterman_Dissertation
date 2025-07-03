import pandas as pd

# 1. 读取数据
df = pd.read_csv("apple_only_grouped.csv")

# 2. 按公司和季度去重求平均：先做公司-季度的平均值
grouped = df.groupby(["FILINGMANAGER_NAME", "REPORTCALENDARORQUARTER"])["VALUE"].sum().reset_index()

# 3. 然后再按公司求“季度平均持仓”
avg_holdings = (
    grouped.groupby("FILINGMANAGER_NAME")["VALUE"]
    .mean()
    .sort_values(ascending=False)
    .head(100)
    .index.tolist()
)

# 4. 从原始数据中过滤这些公司
filtered_df = df[df["FILINGMANAGER_NAME"].isin(avg_holdings)]

# 5. 透视表：行是公司，列是季度，值是持仓金额
pivot_df = filtered_df.pivot_table(
    index="FILINGMANAGER_NAME",
    columns="REPORTCALENDARORQUARTER",
    values="VALUE",
    aggfunc="sum"
)

# 6. 按时间排序列
pivot_df.columns = pd.to_datetime(pivot_df.columns, format="%d-%b-%Y", errors="coerce")
pivot_df = pivot_df.sort_index(axis=1)
pivot_df.columns = pivot_df.columns.strftime("%d-%b-%Y")

# 7. 保存结果
pivot_df.to_excel("top_100_avg_holdings.xlsx")