import pandas as pd


# 1. 设置你想要筛选的公司名称列表（可自定义）
target_companies = [
    "BlackRock", 
    "Vanguard", 
    "FSA Wealth Management LLC", 
    "FULLER & THALER ASSET MANAGEMENT, INC.",
    "FIDELITY D & D BANCORP INC"
]

# 2. 读取原始数据
df = pd.read_csv("apple_only_grouped.csv")

# 3. 筛选出目标公司数据
filtered_df = df[df["FILINGMANAGER_NAME"].isin(target_companies)]

# 4. 透视表，将时间作为列，公司名称作为行，持仓金额为值
pivot_df = filtered_df.pivot_table(
    index="FILINGMANAGER_NAME",
    columns="REPORTCALENDARORQUARTER",
    values="VALUE",
    aggfunc="sum"  # 如果同一公司同一时间点有多条记录则求和
)

# 5. 将列按时间顺序排序（如果格式为 DD-MMM-YYYY 则需要先转换）
pivot_df.columns = pd.to_datetime(pivot_df.columns, format="%d-%b-%Y")
pivot_df = pivot_df.sort_index(axis=1)
pivot_df.columns = pivot_df.columns.strftime("%d-%b-%Y")  # 转换回字符串格式

# 6. 保存为新的 Excel 文件
pivot_df.to_excel("selected_companies_holdings.xlsx")