import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ===========================
# Configurable Parameters
# ===========================
target_fund = "180 WEALTH ADVISORS, LLC"      # 基金经理
target_company = "palo_alto"                 # 公司名称
base_path = "Data/Fundamental_Quarterly/"    # 财务数据路径
macro_path = "Data/Macro/Macro.xlsx"         # 宏观数据文件
fund_path = "Data/13F/palo_alto_only_grouped.csv"  # 基金持仓文件
output_path = f"Data/Merged/{target_company}_{target_fund}_merged_financials.csv"

# Lasso + PCA 参数
n_components = 5                             # PCA主成分个数
alphas = np.logspace(-2, 4, 200)             # Lasso搜索的alpha范围
cv_folds = 3                                 # 交叉验证折数
max_iter = 5000                              # 最大迭代次数
random_state = 42                            # 随机种子

# ===========================
# Helper Functions
# ===========================
def fix_headers(df):
    """Transpose dataframe, set first row as header, drop it."""
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    return df

def clean_quarter_column(df):
    """Extract clean quarter labels like 2025(Q3)."""
    def extract_quarter(x):
        match = re.search(r'\d{4}\s*\(Q\d\)', str(x))
        return match.group(0).replace(" ", "") if match else None
    df.iloc[:, 0] = df.iloc[:, 0].apply(extract_quarter)
    return df

def to_quarter_label(dt):
    """Convert datetime to string like 2025(Q3)."""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}(Q{q})"

# ===========================
# Step 0: get the information of target_fund and target_company
# ===========================
funds = pd.read_csv("long_term_fund_stats_all_tickers.csv")

# ===========================
# Step 1: Load and preprocess macro data
# ===========================
macro_df = pd.read_excel(macro_path).iloc[:21]
macro_df['Date'] = pd.to_datetime(macro_df['Date'], format='%m/%d/%y')
macro_df['YearMonth'] = macro_df['Date'].dt.to_period('M').dt.to_timestamp()

# ===========================
# Step 2: Load and preprocess fund data
# ===========================
palo_alto_df = pd.read_csv(fund_path)
fund_data = palo_alto_df[palo_alto_df['FILINGMANAGER_NAME'] == target_fund].copy()
fund_data['REPORTCALENDARORQUARTER'] = pd.to_datetime(
    fund_data['REPORTCALENDARORQUARTER'], format='%d-%b-%Y'
)
fund_data = fund_data.sort_values(by='REPORTCALENDARORQUARTER').reset_index(drop=True)
fund_data['YearMonth'] = fund_data['REPORTCALENDARORQUARTER'].dt.to_period('M').dt.to_timestamp()

# Merge fund with macro
merged_df = pd.merge(fund_data, macro_df, on='YearMonth', how='inner').drop(columns=['Date'])
merged_df['QuarterLabel'] = merged_df['YearMonth'].apply(to_quarter_label)

# ===========================
# Step 3: Load and preprocess financial statements
# ===========================
bs_df = fix_headers(pd.read_excel(f"{base_path}{target_company}_BS_Quarterly.xlsx", header=None).T)
cfs_df = fix_headers(pd.read_excel(f"{base_path}{target_company}_CFS_Quarterly.xlsx", header=None).T)
is_df = fix_headers(pd.read_excel(f"{base_path}{target_company}_IS_Quarterly.xlsx", header=None).T)

bs_df = clean_quarter_column(bs_df)
cfs_df = clean_quarter_column(cfs_df).drop(index=0).reset_index(drop=True)
is_df = clean_quarter_column(is_df).drop(index=0).reset_index(drop=True)

# Rename first column to QuarterLabel
for df_ in [bs_df, cfs_df, is_df]:
    df_.rename(columns={df_.columns[0]: 'QuarterLabel'}, inplace=True)

# ===========================
# Step 4: Merge all financials with fund+macro data
# ===========================
merged_financials = (
    merged_df
    .merge(bs_df, on='QuarterLabel', how='left')
    .merge(cfs_df, on='QuarterLabel', how='left')
    .merge(is_df, on='QuarterLabel', how='left')
)
# Drop any "Report Filing" columns
cols_to_drop = [col for col in merged_financials.columns if col.startswith("Report Filing")]
merged_financials = merged_financials.drop(columns=cols_to_drop)
# Save merged data
merged_financials.to_csv(output_path, index=False)

# ===========================
# Step 5: PCA + Lasso modeling
# ===========================
df = merged_financials.copy()
y = df['VALUE'].values
# 只保留数值型特征
X = df.drop(columns=['VALUE', 'REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME', 'QuarterLabel'])
X = X.select_dtypes(include=[np.number]).values

# Pipeline: 标准化 → PCA → LassoCV
model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=min(n_components, X.shape[1]))),
    ('lasso', LassoCV(alphas=alphas, cv=cv_folds, max_iter=max_iter, random_state=random_state))
])
model.fit(X, y)

# ===========================
# Step 6: Results
# ===========================
lasso = model.named_steps['lasso']
pca = model.named_steps['pca']
y_pred = model.predict(X)

print("Best alpha:", lasso.alpha_)
print("Explained variance by PCA:", pca.explained_variance_ratio_)
print("\nLasso coefficients on principal components:")
for i, coef in enumerate(lasso.coef_):
    print(f"PC{i+1}: {coef:.4f}")

print("\nIn-sample predictions:", y_pred)
print("R2:", r2_score(y, y_pred))