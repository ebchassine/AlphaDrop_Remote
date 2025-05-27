"""
Read WW layer summary and generate key diagnostic plots.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "weightwatcher_analysis/ww_layer_summary.csv"
OUT_DIR   = "weightwatcher_analysis/graphs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# 1) Power-Law exponent per layer
plt.figure(figsize=(8,4))
plt.plot(df["layer_index"], df["alpha"], marker="o")
plt.title("WW: Power-Law α per Layer")
plt.xlabel("Layer Index")
plt.ylabel("α (heavy-tail exponent)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "alpha_per_layer.png"))

# 2) Spectral norm per layer
plt.figure(figsize=(8,4))
plt.plot(df["layer_index"], df["spectral_norm"], marker="o", color="tab:orange")
plt.title("WW: Spectral Norm λmax per Layer")
plt.xlabel("Layer Index")
plt.ylabel("λmax")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spectral_norm_per_layer.png"))

print(f"✅ Plots saved to {OUT_DIR}/")
