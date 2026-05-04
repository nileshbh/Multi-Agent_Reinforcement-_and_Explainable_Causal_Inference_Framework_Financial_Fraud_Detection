# step1_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. LOAD DATA ----------
df = pd.read_csv("creditcard.csv")      # <-- put your CSV file name here
print("\n Dataset Loaded Successfully")
print(df.head())

# ---------- 2. BASIC INFO ----------
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum().sum())

# ---------- 3. TRANSFORM AMOUNT (Eq. 1) ----------
df["Amount_log"] = np.log1p(df["Amount"])   # log(1 + Amount)

# ---------- 4. CONVERT TIME INTO CYCLIC COMPONENTS (Eq. 2) ----------
df["Time_sin"] = np.sin(2 * np.pi * (df["Time"] % 86400) / 86400)
df["Time_cos"] = np.cos(2 * np.pi * (df["Time"] % 86400) / 86400)

# ---------- 5. SCALE NUMERIC FEATURES ----------
cols_to_scale = [c for c in df.columns if c not in ["Class"]]
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

print("\n Scaling Completed. Displaying sample rows:")
print(df_scaled.head())

# ---------- 6. PLOT DISTRIBUTIONS ----------
plt.figure(figsize=(8,4))
sns.histplot(df["Amount"], bins=30, kde=True, color='gray', label='Original')
sns.histplot(df["Amount_log"], bins=30, kde=True, color='orange', label='Log')
plt.legend(); plt.title("Distribution Before vs After Log Transform")
plt.show()

# ---------- 7. CORRELATION HEATMAP ----------
plt.figure(figsize=(10,6))
sns.heatmap(df_scaled.corr().iloc[-4:, :5], cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (last 4 vs first 5 features)")
plt.show()

# ---------- 8. PRINT BASIC STATISTICS ----------
print("\nDescriptive Stats of Transformed Columns:")
print(df_scaled[['Amount_log','Time_sin','Time_cos']].describe())

# ---------- 9. SAVE FOR NEXT STEP ----------
df_scaled.to_csv("processed_data.csv", index=False)
print("\n Preprocessed file saved as processed_data.csv")
