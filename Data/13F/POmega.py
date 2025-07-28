import pandas as pd
import numpy as np

df = pd.read_csv("hedge_fund_impact_summary.csv", index_col=0)

# 提取所有 tickers
tickers = sorted({col.split("_")[-1] for col in df.columns if col.startswith("Score_")})

# 构建 P：用 Quarter_Count × Avg_Holdings
P_raw = pd.DataFrame(index=df.index)
for t in tickers:
    q_col = f"Quarter_Count_{t}"
    a_col = f"Average_Holdings_Per_Quarter_{t}"
    if q_col in df.columns and a_col in df.columns:
        P_raw[t] = df[q_col] * df[a_col]

# Normalize each row (基金观点权重)
P = P_raw.div(P_raw.sum(axis=1), axis=0).fillna(0)

# 构建 Omega：根据 Score 总值反比（score 越大越 confident）
score_cols = [f"Score_{t}" for t in tickers]
score_sum = df[score_cols].sum(axis=1)
omega_diag = 1 / (score_sum.abs() + 1e-6)  # 避免除零，取 abs 防止负分报错

# 转成对角矩阵
Omega = np.diag(omega_diag)

# 保存
P.to_csv("P_matrix.csv")
pd.DataFrame(Omega, index=df.index, columns=df.index).to_csv("Omega_matrix.csv")
print("✅ 已生成 P_matrix.csv 和 Omega_matrix.csv")