import pandas as pd
import os
# ======================
# 🔧 可配置参数（修改这些即可适用于其他公司）
# ======================
target_keywords = ['apple']  # 核心关键词，可改为 ['microsoft'], ['tesla'] 等
target_identifiers = ['inc', 'inc.', 'incorporated', 'computer', ', inc', 'apple inc']  # 公司名称限定
exclude_keywords = ['pineapple', 'hospitality', 'reit', 'pepper', 'put']  # 排除误匹配项

# 自动根据 target_keywords 设置前缀名（取第一个关键词作为主名）
prefix = target_keywords[0].lower().replace(" ", "_")

# ======================
# 📁 数据读取
# ======================


# 🔧 路径设置：包含所有季度子文件夹的根目录
base_dir = "."  # 当前目录下的所有 form13f 文件夹

# 所有子目录（只要包含 *_form13f 或形如 2024DEC2025Feb 的）
form13f_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and "form13f" in d.lower() or "2024dec2025feb" in d.lower()]

# 初始化总表
info_all = []
cover_all = []

# 遍历所有季度文件夹并读取文件
for folder in form13f_dirs:
    info_path = os.path.join(base_dir, folder, "INFOTABLE.tsv")
    cover_path = os.path.join(base_dir, folder, "COVERPAGE.tsv")
    
    if os.path.exists(info_path) and os.path.exists(cover_path):
        try:
            info_df = pd.read_csv(info_path, sep="\t")
            cover_df = pd.read_csv(cover_path, sep="\t")
            
            info_all.append(info_df)
            cover_all.append(cover_df)
            print(f"✅ Loaded: {folder}")
        except Exception as e:
            print(f"⚠️ Failed to load {folder}: {e}")

# 合并成总数据表
info = pd.concat(info_all, ignore_index=True)
cover = pd.concat(cover_all, ignore_index=True)

print("📦 合并完成：所有季度 INFOTABLE 和 COVERPAGE")


# 🔍 过滤目标公司相关记录（根据关键词）
df = info.copy()


df['issuer_lower'] = df['NAMEOFISSUER'].str.lower()

# 包含目标关键词
for kw in target_keywords:
    df = df[df['issuer_lower'].str.contains(kw, na=False)]

# 同时包含公司标识字段（inc / incorporated 等）
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

# ✅ 保留 INVESTMENTDISCRETION 为 SOLE 的行
df = df[df['INVESTMENTDISCRETION'] == 'SOLE']

# ======================
# 🗂️ 数据清理 & 导出完整行
# ======================
df['REPORTCALENDARORQUARTER_dt'] = pd.to_datetime(
    df['REPORTCALENDARORQUARTER'], format='%d-%b-%Y'
)
df = df.sort_values(by='REPORTCALENDARORQUARTER_dt', ascending=False)
# 清洗后的完整文件保存
df.to_csv(f"{prefix}_only_cleaned.csv", index=False)
print(f"✅ 已将清洗后的结果保存为 {prefix}_only_cleaned.csv")

# ======================
# 📊 分组聚合持仓数据
# ======================
selected_df = df[['VALUE', 'REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME']].copy()
grouped_df = selected_df.groupby(
    ['REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME'], as_index=False
)['VALUE'].sum()

grouped_df['REPORTCALENDARORQUARTER_dt'] = pd.to_datetime(
    grouped_df['REPORTCALENDARORQUARTER'], format='%d-%b-%Y'
)
grouped_df = grouped_df.sort_values(by='REPORTCALENDARORQUARTER_dt', ascending=False)
grouped_df = grouped_df.drop(columns=['REPORTCALENDARORQUARTER_dt'])

# 分组聚合结果保存
grouped_df.to_csv(f"{prefix}_only_grouped.csv", index=False)
print(f"✅ 已将分组结果保存为 {prefix}_only_grouped.csv")