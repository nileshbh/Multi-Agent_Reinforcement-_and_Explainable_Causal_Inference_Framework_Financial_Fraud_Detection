# step3_causal_inference.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

# ---------- 1. LOAD ENGINEERED DATA ----------
df = pd.read_csv("engineered_data.csv")
print("\n Engineered dataset loaded:", df.shape)

# ---------- 2. SELECT FEATURES FOR CAUSAL DISCOVERY ----------
# Using subset to keep graph interpretable (adjust if needed)
selected_cols = ['Amount_log', 'AvgAmount_t', 'Freq_t', 'Lag_Time', 'Class']
data = df[selected_cols].values

# ---------- 3. SPLIT INTO 4 TEMPORAL ENVIRONMENTS ----------
env_splits = np.array_split(df, 4)
print("\n Data split into 4 temporal environments (E1–E4):")
for i, env in enumerate(env_splits, start=1):
    print(f"Environment E{i}: {env.shape}")

# ---------- 4. RUN PC ALGORITHM FOR EACH ENVIRONMENT ----------
graphs = []
for i, env in enumerate(env_splits, start=1):
    print(f"\n🔹 Building causal graph for E{i} ...")
    cg = pc(env[selected_cols].values, alpha=0.05)
    graphs.append(cg)
    GraphUtils.to_pydot(cg.G).write_png(f"causal_graph_E{i}.png")
    print(f" Saved → causal_graph_E{i}.png")

# ---------- 5. PICK ONE GRAPH FOR VISUALIZATION (Fig 3a) ----------
G = graphs[0].G
plt.figure(figsize=(5, 4))
nx_G = nx.DiGraph()
for i, node in enumerate(selected_cols):
    nx_G.add_node(node)
for (i, j) in np.argwhere(G.graph != 0):
    nx_G.add_edge(selected_cols[i], selected_cols[j])
nx.draw(nx_G, with_labels=True, node_color="lightblue",
        font_size=9, node_size=2200, edge_color="gray", arrows=True)
plt.title("Causal Graph in Fraud Case (Fig 3a)")
plt.tight_layout()
plt.show()

# ---------- 6. MODIFY GRAPH (Fig 3b – Removed Causal Link) ----------
G_mod = nx_G.copy()
if ("Freq_t", "Class") in G_mod.edges:
    G_mod.remove_edge("Freq_t", "Class")
plt.figure(figsize=(5, 4))
nx.draw(G_mod, with_labels=True, node_color="salmon",
        font_size=9, node_size=2200, edge_color="black", arrows=True)
plt.title("Modified Causal Graph (Fig 3b)")
plt.tight_layout()
plt.show()

# ---------- 7. SAVE ADJACENCY MATRIX ----------
adj_matrix = nx.to_pandas_adjacency(nx_G)
adj_matrix.to_csv("causal_adjacency_E1.csv", index=True)
print("\n Adjacency matrix saved → causal_adjacency_E1.csv")

print("\n Step 3 Causal Inference completed successfully!")
