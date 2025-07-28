import pandas as pd
import re

# Define file paths
files = {
    'AAPL': 'AAPL_BS.xlsx',
    'AMD': 'AMD_BS.xlsx',
    'AMZN': 'AMZN_BS.xlsx',
    'ASML': 'ASML_BS.xlsx',
    'CRM': 'CRM_BS.xlsx',
    'PANW': 'PANW_BS.xlsx',
    'PLTR': 'PLTR_BS.xlsx',
    'SHOP': 'SHOP_BS.xlsx',
    'SNOW':'SNOW_BS.xlsx',
    'GOOGL': 'GOOGL_BS.xlsx',
    'META': 'META_BS.xlsx',
    'MSFT': 'MSFT_BS.xlsx',
    'NVDA': 'NVDA_BS.xlsx',
    'TSLA': 'TSLA_BS.xlsx'
}

# Function to clean and standardize column names
def standardize_columns(df):
    def extract_year(col):
        match = re.search(r'(20\d{2}|19\d{2})', col)
        return match.group(1) if match else col.strip()
    
    df.columns = [extract_year(col) for col in df.columns]
    return df

# Read, clean, label and collect DataFrames
dfs = []
for ticker, filepath in files.items():
    df = pd.read_excel(filepath)
    df = standardize_columns(df)
    df.insert(0, 'Ticker', ticker)  # Add a column to identify the company
    dfs.append(df)

# Merge all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Save to Excel
merged_df.to_excel("merged_BS.xlsx", index=False)