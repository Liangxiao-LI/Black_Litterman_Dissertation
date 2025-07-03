# analyze_company.py

import pandas as pd
import os


# 前缀用于输出文件命名
prefix = target_keywords[0].lower().replace(" ", "_")

# ✅ 加载缓存
info = pd.read_pickle("all_info_combined.pkl")
cover = pd.read_pickle("all_cover_combined.pkl")

# 🔍 筛选公司记录
df = info.copy()

df['issuer_lower'] = df['NAMEOFISSUER'].str.lower()

temp = df

# ✅ 修改这里：目标公司关键词配置
target_keywords = ['apple']
target_identifiers = ['inc', 'inc.', 'incorporated', 'computer', ', inc', 'apple inc']
exclude_keywords = ['pineapple', 'hospitality', 'reit', 'pepper', 'put']


df = temp

# 包含目标关键词
for kw in target_keywords:
    df = df[df['issuer_lower'].str.contains(kw, na=False)]

# 同时包含公司标识字段
identifier_pattern = '|'.join(target_identifiers)
df = df[df['issuer_lower'].str.contains(identifier_pattern, na=False)]

# 排除无关关键词
for kw in exclude_keywords:
    df = df[~df['issuer_lower'].str.contains(kw, na=False)]

# 合并时间信息
df = df.merge(
    cover[['ACCESSION_NUMBER', 'FILINGMANAGER_NAME', 'REPORTCALENDARORQUARTER', 'DATEREPORTED']],
    on='ACCESSION_NUMBER',
    how='left'
)

# 只保留 SOLE 持仓
df = df[df['INVESTMENTDISCRETION'] == 'SOLE']

# 排序时间字段
df['REPORTCALENDARORQUARTER_dt'] = pd.to_datetime(
    df['REPORTCALENDARORQUARTER'], format='%d-%b-%Y'
)
df = df.sort_values(by='REPORTCALENDARORQUARTER_dt', ascending=False)

# 输出清洗后完整数据
#df.to_csv(f"{prefix}_only_cleaned.csv", index=False)
#print(f"✅ 已保存清洗后数据为：{prefix}_only_cleaned.csv")

# 分组聚合
selected_df = df[['VALUE', 'REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME']].copy()
grouped_df = selected_df.groupby(
    ['REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME'], as_index=False
)['VALUE'].sum()

grouped_df['REPORTCALENDARORQUARTER_dt'] = pd.to_datetime(
    grouped_df['REPORTCALENDARORQUARTER'], format='%d-%b-%Y'
)
grouped_df = grouped_df.sort_values(by='REPORTCALENDARORQUARTER_dt', ascending=False)
grouped_df = grouped_df.drop(columns=['REPORTCALENDARORQUARTER_dt'])

# 输出聚合后数据
grouped_df.to_csv(f"{prefix}_only_grouped.csv", index=False)
print(f"✅ 已保存分组后数据为：{prefix}_only_grouped.csv")