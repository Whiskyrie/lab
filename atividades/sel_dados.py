"""Feature Selection: Filter and Wrapper Methods"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Get the directory where this script is located
script_dir = Path(__file__).parent
data = pd.read_csv(script_dir / "winequality-red.csv", sep=";")

X = data.drop("quality", axis=1).values
y = data["quality"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (filter)
selector_filtro = SelectKBest(score_func=f_classif, k=5)
X_filtro = selector_filtro.fit_transform(X_scaled, y)

# Get selected features from filter method
features_filtro = selector_filtro.get_support(indices=True)
feature_names = data.drop("quality", axis=1).columns
print("=" * 60)
print("MÉTODO DE FILTRO (SelectKBest)")
print("=" * 60)
print(f"Atributos selecionados: {list(feature_names[features_filtro])}")
print(f"Índices: {list(features_filtro)}")

# Feature selection (wrapper)
logreg = LogisticRegression(max_iter=5000, solver="lbfgs")
selector_wrapper = RFE(logreg, n_features_to_select=5)
X_wrapper = selector_wrapper.fit_transform(X_scaled, y)

# Get selected features from wrapper method
features_wrapper = selector_wrapper.get_support(indices=True)
print("\n" + "=" * 60)
print("MÉTODO WRAPPER (RFE)")
print("=" * 60)
print(f"Atributos selecionados: {list(feature_names[features_wrapper])}")
print(f"Índices: {list(features_wrapper)}")

# Compare selected features
features_comuns = set(features_filtro) & set(features_wrapper)
print("\n" + "=" * 60)
print("COMPARAÇÃO DOS MÉTODOS")
print("=" * 60)
print(f"Atributos em comum: {len(features_comuns)} de 5")
if features_comuns:
    print(f"Nomes: {list(feature_names[list(features_comuns)])}")

# Evaluate
acc_filtro = cross_val_score(logreg, X_filtro, y, cv=5).mean()
acc_wrapper = cross_val_score(logreg, X_wrapper, y, cv=5).mean()

print("\n" + "=" * 60)
print("DESEMPENHO DOS MODELOS")
print("=" * 60)
print(f"Acurácia com método de Filtro:  {acc_filtro:.4f}")
print(f"Acurácia com método Wrapper:    {acc_wrapper:.4f}")
print(f"Diferença absoluta:              {abs(acc_filtro - acc_wrapper):.4f}")
print(
    f"Diferença relativa:              {abs(acc_filtro - acc_wrapper) / max(acc_filtro, acc_wrapper) * 100:.2f}%"
)

# Statistical significance check (simple comparison)
if abs(acc_filtro - acc_wrapper) < 0.01:
    print("\nConclusão: Diferença NÃO significativa (< 1%)")
elif abs(acc_filtro - acc_wrapper) < 0.05:
    print("\nConclusão: Diferença MODERADA (1-5%)")
else:
    print("\nConclusão: Diferença SIGNIFICATIVA (> 5%)")

print("=" * 60)
