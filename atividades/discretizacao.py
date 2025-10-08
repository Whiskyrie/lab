"""
Discretização com Tabelas: Uniform vs Quantile
Dataset: Wine Quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# Carregar dados
df = pd.read_csv("atividades/winequality-red.csv", sep=";")
X = df[["alcohol"]].values

print("=" * 70)
print("DISCRETIZAÇÃO DO TEOR ALCOÓLICO - WINE QUALITY DATASET")
print("=" * 70)

# Estatísticas originais
print("\n📊 TABELA 1: DADOS ORIGINAIS")
print("-" * 70)
original_stats = pd.DataFrame(
    {
        "Estatística": ["Mínimo", "Máximo", "Média", "Mediana", "Desvio Padrão"],
        "Valor (%)": [
            f"{X.min():.2f}",
            f"{X.max():.2f}",
            f"{X.mean():.2f}",
            f"{np.median(X):.2f}",
            f"{X.std():.2f}",
        ],
    }
)
print(original_stats.to_string(index=False))

# Discretização
n_bins = 5

# UNIFORMs
uniform_disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
X_uniform = uniform_disc.fit_transform(X).flatten()
uniform_edges = uniform_disc.bin_edges_[0]

print("\n\n📏 TABELA 2: DISCRETIZAÇÃO UNIFORM (MESMA LARGURA)")
print("-" * 70)
uniform_data = []
for i in range(n_bins):
    count = np.sum(X_uniform == i)
    largura = uniform_edges[i + 1] - uniform_edges[i]
    uniform_data.append(
        {
            "Bin": i,
            "Intervalo (%)": f"[{uniform_edges[i]:.2f}, {uniform_edges[i+1]:.2f})",
            "Largura (%)": f"{largura:.2f}",
            "Amostras": count,
            "Porcentagem": f"{(count/len(X)*100):.1f}%",
        }
    )

uniform_table = pd.DataFrame(uniform_data)
print(uniform_table.to_string(index=False))

# QUANTILE
quantile_disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
X_quantile = quantile_disc.fit_transform(X).flatten()
quantile_edges = quantile_disc.bin_edges_[0]

print("\n\n📊 TABELA 3: DISCRETIZAÇÃO QUANTILE (MESMA QUANTIDADE)")
print("-" * 70)
quantile_data = []
for i in range(n_bins):
    count = np.sum(X_quantile == i)
    largura = quantile_edges[i + 1] - quantile_edges[i]
    quantile_data.append(
        {
            "Bin": i,
            "Intervalo (%)": f"[{quantile_edges[i]:.2f}, {quantile_edges[i+1]:.2f})",
            "Largura (%)": f"{largura:.2f}",
            "Amostras": count,
            "Porcentagem": f"{(count/len(X)*100):.1f}%",
        }
    )

quantile_table = pd.DataFrame(quantile_data)
print(quantile_table.to_string(index=False))

# Comparação resumida
print("\n\n📈 TABELA 4: COMPARAÇÃO RESUMIDA")
print("-" * 70)

uniform_counts = np.bincount(X_uniform.astype(int))
quantile_counts = np.bincount(X_quantile.astype(int))

comparison = pd.DataFrame(
    {
        "Estratégia": ["UNIFORM", "QUANTILE"],
        "Largura dos Bins": [
            "Constante (1.30%)",
            f"Variável ({(quantile_edges[1:] - quantile_edges[:-1]).min():.2f}% - {(quantile_edges[1:] - quantile_edges[:-1]).max():.2f}%)",
        ],
        "Amostras por Bin": [
            f"Desbalanceado ({uniform_counts.min()} - {uniform_counts.max()})",
            f"Balanceado ({quantile_counts.min()} - {quantile_counts.max()})",
        ],
        "Desvio Padrão Amostras": [
            f"{uniform_counts.std():.1f}",
            f"{quantile_counts.std():.1f}",
        ],
    }
)
print(comparison.to_string(index=False))

print("\n" + "=" * 70)
print("\n✅ Análise completa com tabelas gerada!")

# ============================================================================
# GERAR HISTOGRAMA COMPARATIVO
# ============================================================================
print("\n📊 Gerando histograma comparativo...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Discretização: Uniform vs Quantile - Teor Alcoólico dos Vinhos", 
             fontsize=16, fontweight="bold")

# === UNIFORM ===
# Histograma original com bordas
ax = axes[0, 0]
ax.hist(X, bins=40, alpha=0.7, color="skyblue", edgecolor="black")
for edge in uniform_edges[1:-1]:
    ax.axvline(edge, color="red", linestyle="--", linewidth=2.5)
ax.set_xlabel("Teor Alcoólico (%)", fontsize=12)
ax.set_ylabel("Frequência", fontsize=12)
ax.set_title("UNIFORM: Intervalos de Mesma Largura", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Distribuição nos bins
ax = axes[0, 1]
bars = ax.bar(range(n_bins), uniform_counts, color="coral", edgecolor="black", linewidth=2)
ax.set_xlabel("Bin", fontsize=12)
ax.set_ylabel("Quantidade de Amostras", fontsize=12)
ax.set_title("UNIFORM: Distribuição nos Bins", fontsize=13, fontweight="bold")
ax.set_xticks(range(n_bins))
for i, (bar, count) in enumerate(zip(bars, uniform_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, count + 15, str(count), 
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# === QUANTILE ===
# Histograma original com bordas
ax = axes[1, 0]
ax.hist(X, bins=40, alpha=0.7, color="lightgreen", edgecolor="black")
for edge in quantile_edges[1:-1]:
    ax.axvline(edge, color="purple", linestyle="--", linewidth=2.5)
ax.set_xlabel("Teor Alcoólico (%)", fontsize=12)
ax.set_ylabel("Frequência", fontsize=12)
ax.set_title("QUANTILE: Intervalos com Mesma Quantidade", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Distribuição nos bins
ax = axes[1, 1]
bars = ax.bar(range(n_bins), quantile_counts, color="orchid", edgecolor="black", linewidth=2)
ax.set_xlabel("Bin", fontsize=12)
ax.set_ylabel("Quantidade de Amostras", fontsize=12)
ax.set_title("QUANTILE: Distribuição nos Bins", fontsize=13, fontweight="bold")
ax.set_xticks(range(n_bins))
for i, (bar, count) in enumerate(zip(bars, quantile_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, count + 15, str(count), 
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("atividades/discretizacao_comparacao.jpg", dpi=300, bbox_inches="tight")
print("✅ Imagem salva: atividades/discretizacao_comparacao.jpg")
