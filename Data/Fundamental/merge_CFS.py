import pandas as pd
import re
import os

# 获取当前目录下所有 CFS 文件
all_files = os.listdir()
cfs_files = [f for f in all_files if f.endswith('_CFS.xlsx')]

# 定义标准化列名为年份的函数
def standardize_columns(df):
    def extract_year(col):
        match = re.search(r'(20\d{2}|19\d{2})', str(col))
        return match.group(1) if match else str(col).strip()
    df.columns = [extract_year(col) for col in df.columns]
    return df

# 读取、清洗并收集 CFS 数据
dfs = []
for filepath in cfs_files:
    try:
        ticker = filepath.split('_')[0]
        df = pd.read_excel(filepath)
        df = standardize_columns(df)
        if 'Ticker' not in df.columns:
            df.insert(0, 'Ticker', ticker)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️  无法处理 {filepath}：{e}")

# 合并并导出
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_excel("merged_CFS.xlsx", index=False)
    print("✅ 合并完成：已保存为 merged_CFS.xlsx")
else:
    print("❌ 没有可用的现金流量表文件进行合并。")
