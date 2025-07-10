import pandas as pd

# 加载数据
info = pd.read_pickle("all_info_combined_newest.pkl")
cover = pd.read_pickle("all_cover_combined_newest.pkl")

# 设置目标基金经理关键词（如 "BLACKROCK", "VANGUARD", "NVIDIA"）
target_keyword = "NVIDIA"

# 筛选 FILINGMANAGER_NAME 中包含该关键词的行
matched_cover = cover[cover['FILINGMANAGER_NAME'].str.contains(target_keyword, case=False, na=False)]
accession_numbers = matched_cover['ACCESSION_NUMBER'].unique()

print(f"✅ 找到 {len(accession_numbers)} 份包含 '{target_keyword}' 的报告。")

# 从 info 中提取对应 ACCESSION_NUMBER 的持仓记录
matched_info = info[info['ACCESSION_NUMBER'].isin(accession_numbers)]

# 选出你关心的字段
columns_to_keep = [
    'NAMEOFISSUER',       # 公司名
    'TITLEOFCLASS',       # 股票类别
    'CUSIP',              # 股票代码
    'VALUE',              # 投资市值
    'SSHPRNAMT',          # 持仓数量（注意有时可能是 SHARES）
    'ACCESSION_NUMBER'    # 追踪用
]

# 处理缺失字段（有些数据可能是 'SHARES' 而不是 'SSHPRNAMT'）
for col in ['VALUE', 'SSHPRNAMT']:
    if col not in matched_info.columns:
        print(f"⚠️ 缺失列: {col}，请检查数据字段名。")

# 筛选存在的列
available_cols = [col for col in columns_to_keep if col in matched_info.columns]
output_df = matched_info[available_cols]

# 打印前几行结果
print(f"\n📊 '{target_keyword}' 持仓信息如下（前10项）：")
print(output_df.head(10))

# 保存完整表格
output_file = f"{target_keyword}_full_holdings.xlsx"
output_df.to_excel(output_file, index=False)
print(f"\n💾 已导出完整持仓信息到：{output_file}")