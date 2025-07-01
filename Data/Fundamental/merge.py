import pandas as pd


# 定义清洗列名的函数
def clean_columns(df):
    return df.columns.str.replace('\n', ' ', regex=False)\
                     .str.replace(r'\s+', ' ', regex=True)\
                     .str.strip()

# 读取三个报表文件
balance_sheet = pd.read_excel("AAPL Balance Sheet Statement (Annual) - Discounting Cash Flows.xlsx")
cash_flow = pd.read_excel("AAPL Cash Flow Statement (Annual) - Discounting Cash Flows.xlsx")
income_statement = pd.read_excel("AAPL Income Statement (Annual) - Discounting Cash Flows (1).xlsx")

# 清洗列名
balance_sheet.columns = clean_columns(balance_sheet)
cash_flow.columns = clean_columns(cash_flow)
income_statement.columns = clean_columns(income_statement)

# 定义目标列（注意空格是单个空格）
target_columns = ["2024 09-28", "2023 09-30", "2022 09-24", "2021 09-25", "2020 09-26"]

# 如果需要保留“项目名称”列（比如第一列），可使用这个方式
def filter_with_index(df):
    first_col = df.columns[0]
    keep_cols = [first_col] + [col for col in target_columns if col in df.columns]
    return df.loc[:, keep_cols]

# 筛选数据
balance_sheet_filtered = filter_with_index(balance_sheet)
cash_flow_filtered = filter_with_index(cash_flow)
income_statement_filtered = filter_with_index(income_statement)

# 写入一个新的 Excel 文件，三个工作表
with pd.ExcelWriter("AAPL_Filtered_Financial_Statements.xlsx") as writer:
    balance_sheet_filtered.to_excel(writer, sheet_name="Balance Sheet", index=False)
    cash_flow_filtered.to_excel(writer, sheet_name="Cash Flow", index=False)
    income_statement_filtered.to_excel(writer, sheet_name="Income Statement", index=False)