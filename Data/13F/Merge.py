import pandas as pd

# 读取数据
info = pd.read_csv("2024DEC2025Feb/INFOTABLE.tsv", sep="\t")
cover = pd.read_csv("2024DEC2025Feb/COVERPAGE.tsv", sep="\t")

# 过滤 Apple 股票
apple = info[info['NAMEOFISSUER'].str.upper().str.contains("APPLE", na=False)]

# 合并并保留时间信息
result = apple.merge(
    cover[['ACCESSION_NUMBER', 'FILINGMANAGER_NAME', 'REPORTCALENDARORQUARTER', 'DATEREPORTED']],
    on='ACCESSION_NUMBER',
    how='left'
)

# 按照 REPORTCALENDARORQUARTER 降序排列，再按照 FILINGMANAGER_NAME 升序排列
result = result.sort_values(
    by=['REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME'],
    ascending=[False, True]
)

# 导出
result.to_csv("apple_holdings_with_date.csv", index=False)