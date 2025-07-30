import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import os
from sklearn.feature_selection import SelectKBest, f_regression


# ===========================
# Configurable Parameters
# ===========================
base_path = "Data/Fundamental_Quarterly/"       # 财务数据路径
macro_path = "Data/Macro/Macro.xlsx"            # 宏观数据文件
funds_path = "long_term_fund_stats_all_tickers.csv"  # 基金列表
fund_data_path_template = "Data/13F/{ticker}_only_grouped.csv"  # 13F路径模板
output_dir = "Data/Merged/"                     # 输出文件夹
os.makedirs(output_dir, exist_ok=True)

# Lasso + PCA 参数
n_components = 5
alphas = np.logspace(-2, 5, 200)
cv_folds = 3
max_iter = 50000
random_state = 42
quarters = [f"{y}(Q{q})" for y in range(2020, 2025) for q in range(1, 5)]
start_index = quarters.index("2021(Q4)")  # 2020Q4 是训练数据截止点

# ===========================
# Helper Functions
# ===========================
def fix_headers(df):
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    return df

def clean_quarter_column(df):
    def extract_quarter(x):
        match = re.search(r'\d{4}\s*\(Q\d\)', str(x))
        return match.group(0).replace(" ", "") if match else None
    df.iloc[:, 0] = df.iloc[:, 0].apply(extract_quarter)
    return df

def to_quarter_label(dt):
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}(Q{q})"

def safe_filename(name: str) -> str:
    # 替换掉不安全的文件名字符
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def next_quarter(q):
    year, qtr = int(q[:4]), int(q[-2])
    if qtr == 4:
        return f"{year+1}(Q1)"
    else:
        return f"{year}(Q{qtr+1})"

def safe_parse_date(x):
    try:
        return pd.to_datetime(x, format='%d-%b-%Y')
    except Exception:
        return x  # 如果是 2024(Q2) 这种季度格式，直接返回原值

def prefilter_funds(funds, cutoff_quarter, min_points=3):
    valid_pairs = []
    for _, row in funds.iterrows():
        fund_name, ticker = row['FILINGMANAGER_NAME'], row['ticker']
        fund_path = fund_data_path_template.format(ticker=ticker)
        if not os.path.exists(fund_path):
            continue
        # 1. 加载基金数据
        fund_df = pd.read_csv(fund_path)
        fund_df = fund_df[fund_df['FILINGMANAGER_NAME'] == fund_name]
        if fund_df.empty:
            continue
        fund_df['REPORTCALENDARORQUARTER'] = fund_df['REPORTCALENDARORQUARTER'].apply(safe_parse_date)
        fund_df['YearMonth'] = fund_df['REPORTCALENDARORQUARTER'].dt.to_period('M').dt.to_timestamp()
        fund_df['QuarterLabel'] = fund_df['REPORTCALENDARORQUARTER'].apply(lambda d: f"{d.year}(Q{(d.month-1)//3+1})")
        fund_df = fund_df[fund_df['QuarterLabel'] <= cutoff_quarter]
        if fund_df.empty:
            continue
        # 2. 加载宏观 & 财报（和 process_one_fund 一致）
        macro_df = pd.read_excel(macro_path).iloc[:21]
        macro_df['Date'] = pd.to_datetime(macro_df['Date'], format='%m/%d/%y')
        macro_df['YearMonth'] = macro_df['Date'].dt.to_period('M').dt.to_timestamp()
        merged_df = pd.merge(fund_df, macro_df, on='YearMonth', how='inner').drop(columns=['Date'])
        merged_df['QuarterLabel'] = merged_df['YearMonth'].apply(to_quarter_label)
        try:
            bs_df = fix_headers(pd.read_excel(f"{base_path}{ticker}_BS_Quarterly.xlsx", header=None).T)
            cfs_df = fix_headers(pd.read_excel(f"{base_path}{ticker}_CFS_Quarterly.xlsx", header=None).T)
            is_df = fix_headers(pd.read_excel(f"{base_path}{ticker}_IS_Quarterly.xlsx", header=None).T)
            bs_df = clean_quarter_column(bs_df)
            cfs_df = clean_quarter_column(cfs_df).drop(index=0).reset_index(drop=True)
            is_df = clean_quarter_column(is_df).drop(index=0).reset_index(drop=True)
            for df_ in [bs_df, cfs_df, is_df]:
                df_.rename(columns={df_.columns[0]: 'QuarterLabel'}, inplace=True)
            merged_financials = (
                merged_df
                .merge(bs_df, on='QuarterLabel', how='left')
                .merge(cfs_df, on='QuarterLabel', how='left')
                .merge(is_df, on='QuarterLabel', how='left')
            )
        except Exception:
            continue
        # 3. 样本量判断
        if len(merged_financials) >= min_points:
            valid_pairs.append((fund_name, ticker))
    return pd.DataFrame(valid_pairs, columns=['FILINGMANAGER_NAME','ticker'])

def process_one_fund(target_fund, target_company, cutoff_quarter):
    try:
        # ---- Step 1: Load macro data ----
        macro_df = pd.read_excel(macro_path).iloc[:21]
        macro_df['Date'] = pd.to_datetime(macro_df['Date'], format='%m/%d/%y')
        macro_df['YearMonth'] = macro_df['Date'].dt.to_period('M').dt.to_timestamp()

        # ---- Step 2: Load fund data ----
        fund_path = fund_data_path_template.format(ticker=target_company)
        if not os.path.exists(fund_path):
            print(f"⚠️ Skipped {target_fund} - {target_company}: Fund data not found")
            return None
        palo_alto_df = pd.read_csv(fund_path)
        fund_data = palo_alto_df[palo_alto_df['FILINGMANAGER_NAME'] == target_fund].copy()
        if fund_data.empty:
            print(f"⚠️ Skipped {target_fund} - {target_company}: No data for this manager")
            return None
        fund_data['REPORTCALENDARORQUARTER'] = fund_data['REPORTCALENDARORQUARTER'].apply(safe_parse_date)
        fund_data = fund_data.sort_values(by='REPORTCALENDARORQUARTER').reset_index(drop=True)
        fund_data['YearMonth'] = fund_data['REPORTCALENDARORQUARTER'].dt.to_period('M').dt.to_timestamp()
        merged_df = pd.merge(fund_data, macro_df, on='YearMonth', how='inner').drop(columns=['Date'])
        merged_df['QuarterLabel'] = merged_df['YearMonth'].apply(to_quarter_label)

        # ---- Step 3: Load financial statements ----
        bs_df = fix_headers(pd.read_excel(f"{base_path}{target_company}_BS_Quarterly.xlsx", header=None).T)
        cfs_df = fix_headers(pd.read_excel(f"{base_path}{target_company}_CFS_Quarterly.xlsx", header=None).T)
        is_df = fix_headers(pd.read_excel(f"{base_path}{target_company}_IS_Quarterly.xlsx", header=None).T)
        bs_df = clean_quarter_column(bs_df)
        cfs_df = clean_quarter_column(cfs_df).drop(index=0).reset_index(drop=True)
        is_df = clean_quarter_column(is_df).drop(index=0).reset_index(drop=True)
        for df_ in [bs_df, cfs_df, is_df]:
            df_.rename(columns={df_.columns[0]: 'QuarterLabel'}, inplace=True)

        # ---- Step 4: Merge ----
        merged_financials = (
            merged_df
            .merge(bs_df, on='QuarterLabel', how='left')
            .merge(cfs_df, on='QuarterLabel', how='left')
            .merge(is_df, on='QuarterLabel', how='left')
        )
        cols_to_drop = [col for col in merged_financials.columns if col.startswith("Report Filing")]
        merged_financials = merged_financials.drop(columns=cols_to_drop)

        full_financials = merged_financials.copy()

        # ---- Step 4.1: 取最近 8 个季度作为训练集 ----
        all_quarters_sorted = sorted(merged_financials['QuarterLabel'].unique(), key=lambda x: (int(x[:4]), int(x[-2])))
        if cutoff_quarter not in all_quarters_sorted:
            print(f"⚠️ Skipped {target_fund} - {target_company}: No data before {cutoff_quarter}")
            return None
        cutoff_idx = all_quarters_sorted.index(cutoff_quarter)
        train_start_idx = max(0, cutoff_idx - 7)  # 最近 8 个季度
        train_quarters = all_quarters_sorted[train_start_idx:cutoff_idx + 1]
        merged_financials = merged_financials[merged_financials['QuarterLabel'].isin(train_quarters)]

        if merged_financials.empty:
            print(f"⚠️ Skipped {target_fund} - {target_company}: No data before {cutoff_quarter}")
            return None

        safe_fund = safe_filename(target_fund)
        safe_company = safe_filename(target_company)

        output_path = os.path.join(output_dir, f"{safe_company}_{safe_fund}_merged_financials.csv")
        merged_financials.to_csv(output_path, index=False)

        # ---- Step 5: PCA + Lasso ----
        df = merged_financials.copy()
        y = df['VALUE'].values
        X = df.drop(columns=['VALUE', 'REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME', 'QuarterLabel'])
        X = X.select_dtypes(include=[np.number]).values
        if X.shape[0] < 3:
            print(f"⚠️ Skipped {target_fund} - {target_company}: Not enough data points")
            return None

        pca_components = min(n_components, 20, X.shape[0], X.shape[1])  # 样本数和特征数的最小值
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_select', SelectKBest(f_regression, k=min(20, X.shape[1]))),
            ('pca', PCA(n_components=pca_components)),
            ('lasso', LassoCV(alphas=alphas, cv=cv_folds, max_iter=max_iter, random_state=random_state))
        ])

        model.fit(X, y)
        lasso = model.named_steps['lasso']
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        if r2 <= 0:
            print(f"⚠️ {target_fund}-{target_company}: R²={r2:.4f}, switching to PLS...")
            from sklearn.cross_decomposition import PLSRegression
            pls = PLSRegression(n_components=min(3, X.shape[1]))
            pls.fit(X, y)
            y_pred = pls.predict(X)
            r2 = r2_score(y, y_pred)
            print(f"🔄 PLS替代模型: {target_fund}-{target_company}, R²={r2:.4f}")
        print(f"✅ {target_fund} - {target_company}: R2 = {r2:.4f}, Best alpha = {lasso.alpha_:.2f}")

        # ---- Step 6: 预测下一个季度 ----
        future_quarter = next_quarter(cutoff_quarter)
        future_data = full_financials[full_financials['QuarterLabel'] == future_quarter]
        if not future_data.empty:
            future_X = future_data.drop(columns=['VALUE', 'REPORTCALENDARORQUARTER', 'FILINGMANAGER_NAME', 'QuarterLabel'])
            future_X = future_X.select_dtypes(include=[np.number]).values
            future_pred = model.predict(future_X)
        else:
            future_pred = []
        
        prediction_output = os.path.join(output_dir, f"predict_{future_quarter}.xlsx")
        pd.DataFrame({
            "fund": target_fund,
            "ticker": target_company,
            "predicted_value": future_pred
        }).to_excel(prediction_output, index=False)

        return {"fund": target_fund, "ticker": target_company, "r2": r2, "alpha": lasso.alpha_,"future_quarter": future_quarter,"future_pred": future_pred.tolist() if len(future_pred) > 0 else None}

    except Exception as e:
        print(f"❌ Error processing {target_fund} - {target_company}: {e}")
        return None

# ===========================
# Batch Processing
# ===========================
funds_all = pd.read_csv(funds_path)
results = []

for i in range(start_index+1, len(quarters)):  
    cutoff_quarter = quarters[i-1]
    train_start = max(0, i - 8)  # 8 个季度窗口
    train_quarters = quarters[train_start:i]  # 实际训练区间
    print(f"=== Predicting {quarters[i]} using data from {train_quarters[0]} to {train_quarters[-1]} ===")
    funds = prefilter_funds(funds_all, cutoff_quarter, min_points=3)
    for _, row in funds.iterrows():
        res = process_one_fund(row['FILINGMANAGER_NAME'], row['ticker'], cutoff_quarter)
        if res:
            results.append(res)
pd.DataFrame(results).to_csv("modeling_results_summary.csv", index=False)










#Draw the diagram for the timeplot for the 
import matplotlib.pyplot as plt

# 画滚动训练-预测时间线
rolling_windows = []
window_size = 8
for i in range(start_index + 1, len(quarters)):
    train_start = max(0,i-window_size)
    train_end = i - 1
    predict = i
    rolling_windows.append((train_start, train_end, predict))

fig, ax = plt.subplots(figsize=(14, 4))
for idx, (train_start, train_end, predict) in enumerate(rolling_windows):
    ax.barh(idx, train_end - train_start + 1, left=train_start, color='lightgray', edgecolor='black', label='Training' if idx == 0 else "")
    ax.barh(idx, 1, left=predict, color='lightblue', edgecolor='black', label='Prediction' if idx == 0 else "")

ax.set_yticks(range(len(rolling_windows)))
ax.set_yticklabels([f"Step {i+1}" for i in range(len(rolling_windows))])
ax.set_xticks(range(len(quarters)))
ax.set_xticklabels(quarters, rotation=45)
ax.set_xlim(0, len(quarters))
ax.set_title("Rolling Training & Prediction Timeline")
ax.set_xlabel("Quarter")
ax.set_ylabel("Rolling Window Step")
ax.legend()
plt.tight_layout()
plt.show()