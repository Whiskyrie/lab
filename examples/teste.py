"""Comparação de PCA: 2 vs 15 componentes principais no dataset MNIST."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lab.dataset import mnist

# Load and standardize MNIST data
X, y = mnist(1000)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Análise de variância
pca_full = PCA()
pca_full.fit(X_scaled)
variance_explained = pca_full.explained_variance_ratio_
cumulative_variance = variance_explained.cumsum()

print(
    f"\nVariância - 2 comp: {cumulative_variance[1]:.2%} | 15 comp: {cumulative_variance[14]:.2%}"
)

# Configurar estilo dos gráficos
plt.style.use("seaborn-v0_8-darkgrid")
colors = plt.cm.tab10(range(10))

# Comparação lado a lado: 2 vs 15 componentes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor("white")

# PCA com 2 componentes
pca_2 = PCA(n_components=2)
embeddings_2 = pca_2.fit_transform(X_scaled)
var_2 = pca_2.explained_variance_ratio_.sum()

for digit in range(10):
    mask = y == digit
    ax1.scatter(
        embeddings_2[mask, 0],
        embeddings_2[mask, 1],
        c=[colors[digit]],
        label=f"Dígito {digit}",
        alpha=0.7,
        s=60,
        edgecolors="black",
        linewidth=0.5,
    )

ax1.set_title(
    f"PCA - 2 Componentes Principais\nVariância Explicada: {var_2:.2%}",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax1.set_xlabel("Primeira Componente Principal (PC1)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Segunda Componente Principal (PC2)", fontsize=14, fontweight="bold")
ax1.legend(loc="best", frameon=True, shadow=True, fontsize=10, ncol=2)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.tick_params(labelsize=11)

# PCA com 15 componentes (visualizando PC1 vs PC2)
pca_15 = PCA(n_components=15)
embeddings_15 = pca_15.fit_transform(X_scaled)
var_15 = pca_15.explained_variance_ratio_.sum()

for digit in range(10):
    mask = y == digit
    ax2.scatter(
        embeddings_15[mask, 0],
        embeddings_15[mask, 1],
        c=[colors[digit]],
        label=f"Dígito {digit}",
        alpha=0.7,
        s=60,
        edgecolors="black",
        linewidth=0.5,
    )

ax2.set_title(
    f"PCA - 15 Componentes Principais\n(Visualizando PC1 vs PC2)\nVariância Explicada Total: {var_15:.2%}",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax2.set_xlabel("Primeira Componente Principal (PC1)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Segunda Componente Principal (PC2)", fontsize=14, fontweight="bold")
ax2.legend(loc="best", frameon=True, shadow=True, fontsize=10, ncol=2)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(
    "/home/whiskyrie/Projetos/lab/examples/pca_comparison.jpg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"\n✓ Comparação salva em: examples/pca_comparison.jpg")
plt.show()
