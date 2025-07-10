import pandas as pd
import os
import glob

# 1. 查找当前目录下所有 *_only_grouped.csv 文件
csv_files = glob.glob("*_only_grouped.csv")

# 目标起止区间
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("2024-12-31")

for file_path in csv_files:
    # 提取文件前缀（如 apple / google / asml）
    prefix = os.path.basename(file_path).split("_")[0]

    # 读取数据
    df = pd.read_csv(file_path)

    # 将 REPORTCALENDARORQUARTER 转换为 datetime
    df["REPORT_DATE"] = pd.to_datetime(df["REPORTCALENDARORQUARTER"], format="%d-%b-%Y", errors="coerce")

    # 删除解析失败的行
    df = df.dropna(subset=["REPORT_DATE"])

    # 筛选时间区间：2020-01-01 到 2024-12-31
    df = df[(df["REPORT_DATE"] >= start_date) & (df["REPORT_DATE"] <= end_date)]

    # 如果数据为空，跳过
    if df.empty:
        print(f"[跳过] {file_path} 没有符合时间条件的数据。")
        continue

    # 新增一列季度标签：如 "2020Q1", "2020Q2" 等
    df["YEAR"] = df["REPORT_DATE"].dt.year
    df["QUARTER"] = df["REPORT_DATE"].dt.quarter
    df["YEAR_QUARTER"] = df["YEAR"].astype(str) + "Q" + df["QUARTER"].astype(str)

    # 创建透视表：行是 hedge fund，列是季度标签，值是持仓金额
    pivot_df = df.pivot_table(
        index="FILINGMANAGER_NAME",
        columns="YEAR_QUARTER",
        values="VALUE",
        aggfunc="sum"
    ).fillna(0)

    # 按时间顺序排列列
    pivot_df = pivot_df[sorted(pivot_df.columns)]

    # 保存结果
    output_filename = f"{prefix}_fund_matrix.csv"
    pivot_df.to_csv(output_filename)

    print(f"[完成] 保存文件: {output_filename}")

    # 计算季度环比百分比变化
    pct_change_df = pivot_df.pct_change(axis=1) * 100
    pct_change_df = pct_change_df.round(2)

    # 保存百分比变化矩阵
    pct_change_file = f"{prefix}_fund_pct_change_matrix.csv"
    pct_change_df.to_csv(pct_change_file)
    print(f"[完成] 保存持仓变化矩阵: {pct_change_file}")