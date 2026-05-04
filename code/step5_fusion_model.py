# step5_fusion_model.py
import numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
# ---------- 1. SIMULATE PROBABILITIES FROM AGENTS ----------

np.random.seed(42)
p_A = np.random.uniform(0.2, 0.95, 20)   # Q-Learning agent
p_B = np.random.uniform(0.1, 0.9, 20)    # DQN agent
p_C = np.random.uniform(0.3, 0.98, 20)   # Actor–Critic agent

print("\nprobabilities from agents:")
print(pd.DataFrame({"AgentA":p_A,"AgentB":p_B,"AgentC":p_C}).head())

# ---------- 2. DEEPPROBLOG-STYLE BAYESIAN FUSION (Eq. 9) ----------
P_fraud = 0.17
joint_likelihood = P_fraud * p_A * p_B * p_C
P_total = (P_fraud * p_A * p_B * p_C) + ((1-P_fraud)*(1-p_A)*(1-p_B)*(1-p_C))
P_fraud_given_p = joint_likelihood / (P_total + 1e-9)

print("\nPosterior fraud probability (DeepProbLog fusion):")
print(P_fraud_given_p[:5])

# ---------- 3. LOGIC ATTENTION NETWORK (Eq. 10) ----------
# Compute attention weights αᵢ for each agent per transaction
scores = np.vstack([p_A, p_B, p_C]).T
alpha = F.softmax(torch.tensor(scores), dim=1).numpy()

# Compute final weighted output
p_final = np.sum(alpha * scores, axis=1)
print("\nFinal fused fraud probabilities (LAN weighted):")
print(p_final[:5])

# ---------- 4. EXPLAINABILITY – AGENT IMPORTANCE ----------
avg_alpha = alpha.mean(axis=0)
agents = ["Q-Learning","DQN","Actor-Critic"]

plt.figure(figsize=(6,4))
plt.bar(agents, avg_alpha, color=["steelblue","orange","green"])
plt.title("Average Attention Weights per Agent (Explainability)")
plt.ylabel("Attention Weight (α)")
plt.tight_layout(); plt.show()

# ---------- 5. VISUALIZE FUSED FRAUD SCORES ----------
plt.figure(figsize=(8,4))
plt.plot(p_final, marker="o", color="red", label="Fused Fraud Probability")
plt.axhline(0.5, ls="--", c="gray", label="Decision Threshold")
plt.title("Final Fraud Probability after Multi-Agent Fusion")
plt.xlabel("Transaction Index"); plt.ylabel("Fraud Probability")
plt.legend(); plt.tight_layout(); plt.show()

# ---------- 6. DECISION OUTPUT ----------
tau = 0.5
decisions = np.where(p_final >= tau, "Fraud", "Normal")
fraud_ratio = np.sum(decisions=="Fraud") / len(decisions)
print(f"\n Final Decision Threshold = {tau}")
print(f"Predicted Fraudulent Transactions: {fraud_ratio*100:.2f}%")
print("\n Step 5 Multi-Agent Fusion Model completed successfully.")
