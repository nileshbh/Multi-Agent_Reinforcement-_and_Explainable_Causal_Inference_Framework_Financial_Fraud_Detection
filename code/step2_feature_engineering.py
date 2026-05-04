# step2_feature_engineering.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. LOAD PREPROCESSED DATA ----------
df = pd.read_csv("processed_data.csv")
print("\n Processed data loaded:", df.shape)

# ---------- 2. SORT CHRONOLOGICALLY ----------
df = df.sort_values("Time").reset_index(drop=True)

# ---------- 3. CREATE BEHAVIORAL FEATURES ----------
# Eq. 3 – Average amount per time window
window_size = 10                      # ~10 transactions per window
df["AvgAmount_t"] = df["Amount_log"].rolling(window=window_size).mean()

# Eq. 4 – Transaction frequency per time interval
df["Freq_t"] = df["Time"].diff(periods=window_size).apply(lambda x: window_size/x if x != 0 else 0)

# Lag-time difference (ΔTimeᵢ = Timeᵢ – Timeᵢ₋₁)
df["Lag_Time"] = df["Time"].diff().fillna(0)

# Fill any NaN from the first window
df.fillna(0, inplace=True)

print("\n New columns added: AvgAmount_t, Freq_t, Lag_Time")
print(df[["Amount_log", "AvgAmount_t", "Freq_t", "Lag_Time", "Class"]].head(12))

# ---------- 4. SUMMARY STATISTICS ----------
print("\nFeature-engineering summary:")
print(df[["AvgAmount_t", "Freq_t", "Lag_Time"]].describe())

# ---------- 5. VISUALIZATIONS ----------
plt.figure(figsize=(8,4))
sns.lineplot(x=range(300), y=df["AvgAmount_t"][:300])
plt.title("Rolling Average Transaction Amount (Eq. 3)"); plt.xlabel("Transaction Index"); plt.ylabel("AvgAmount_t")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df["Freq_t"], bins=40, kde=True, color="darkcyan")
plt.title("Distribution of Transaction Frequency (Eq. 4)"); plt.xlabel("Freq_t")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df["Lag_Time"], bins=40, kde=True, color="orange")
plt.title("Distribution of Lag Time between Transactions (ΔTimeᵢ)"); plt.xlabel("Lag_Time (sec)")
plt.tight_layout(); plt.show()

# ---------- 6. SAVE ENGINEERED DATA ----------
df.to_csv("engineered_data.csv", index=False)
print("\n Feature-engineered file saved as engineered_data.csv")
