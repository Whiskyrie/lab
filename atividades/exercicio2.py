"""
Exercício 2 - Árvore de Decisão

1. CONJUNTO DE DADOS ESCOLHIDO:
   - Dataset: Iris (sklearn.datasets.load_iris)
   - 150 amostras de flores Iris de 3 espécies diferentes
   - 4 características: comprimento e largura das sépalas e pétalas
   - 3 classes: Setosa, Versicolor, Virginica

2. PROFUNDIDADE DA ÁRVORE (PODA):
   - Fórmula: altura_dela = 1 + último_dígito_de_seu_RA % 4
   - Assumindo último dígito do RA = 8 (ajuste conforme seu RA)
   - Cálculo: 1 + (8 % 4) = 0
   - max_depth = 1

3. PROCESSO DECISÓRIO (exemplo de caminho):
   Caminho 1: Raiz → Folha Esquerda
   - A árvore avalia: "petal width (cm) <= 0.8?"
   - Se SIM: classifica como Setosa
   - Explicação: Flores com pétalas muito estreitas (≤ 0.8 cm) são sempre Setosa

   Caminho 2: Raiz → Folha Direita
   - A árvore avalia: "petal width (cm) <= 0.8?"
   - Se NÃO: classifica como Versicolor/Virginica (necessita mais profundidade para separar)
   - Explicação: Flores com pétalas mais largas (> 0.8 cm) podem ser Versicolor ou Virginica

4. SIGNIFICADO DOS VALORES NAS FOLHAS:
   Cada folha mostra:
   - gini: Índice de impureza de Gini (0 = puro, >0 = misturado)
     * Mede a probabilidade de classificação incorreta
     * gini = 1 - Σ(pi²), onde pi é a proporção de cada classe

   - samples: Número de amostras que chegaram nesta folha
     * Total de exemplos de treinamento que seguiram este caminho

   - value: [#setosa, #versicolor, #virginica]
     * Quantidade de amostras de cada classe nesta folha
     * Exemplo: [50, 0, 0] = 50 Setosa, 0 Versicolor, 0 Virginica

   - class: Classe prevista (maioria)
     * A classe com maior número de amostras nesta folha
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Carregar conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. Criar e treinar árvore com poda
# IMPORTANTE: Ajuste o último dígito do seu RA aqui
ultimo_digito_RA = 8  # <<< ALTERE ESTE VALOR PARA O ÚLTIMO DÍGITO DO SEU RA
max_depth = 1 + (ultimo_digito_RA % 4)

print(f"Último dígito do RA: {ultimo_digito_RA}")
print(f"Profundidade da árvore: max_depth = 1 + ({ultimo_digito_RA} % 4) = {max_depth}")
print()

model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X, y)

# 3. Visualizar a árvore de decisão
plt.figure(figsize=(20, 12))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=12,
)
plt.title(f"Árvore de Decisão - Dataset Iris (max_depth={max_depth})", fontsize=16)
plt.tight_layout()
plt.savefig("arvore_decisao_iris.png", dpi=300, bbox_inches="tight")
print("Árvore salva em: arvore_decisao_iris.png")
plt.show()

# 4. Informações adicionais sobre a árvore
print("\n=== INFORMAÇÕES DA ÁRVORE ===")
print(f"Número de folhas: {model.get_n_leaves()}")
print(f"Profundidade: {model.get_depth()}")
print(f"Acurácia no treino: {model.score(X, y):.4f}")
print(
    f"Feature mais importante: {iris.feature_names[model.feature_importances_.argmax()]}"
)

# 5. EXPLICAÇÃO DETALHADA DO PROCESSO DECISÓRIO
print("\n" + "=" * 70)
print("EXPLICAÇÃO DETALHADA DO PROCESSO DECISÓRIO")
print("=" * 70)

# Obter informações sobre a árvore
tree = model.tree_
feature = tree.feature
threshold = tree.threshold
n_nodes = tree.node_count

print(f"\n📊 A árvore possui {n_nodes} nós no total")
print(f"🌿 Nó raiz usa a feature: {iris.feature_names[feature[0]]}")
print(f"🔢 Limiar de decisão: {threshold[0]:.2f} cm")

# Exemplo de classificação de amostras
print("\n" + "-" * 70)
print("EXEMPLO PRÁTICO - CLASSIFICANDO AMOSTRAS:")
print("-" * 70)

exemplos = [0, 50, 100]  # Índices de exemplos de cada classe
for idx in exemplos:
    amostra = X[idx]
    classe_real = iris.target_names[y[idx]]
    classe_pred = iris.target_names[model.predict([amostra])[0]]

    print(f"\n🌸 Amostra {idx} - Classe real: {classe_real}")
    print(f"   Características:")
    for i, nome in enumerate(iris.feature_names):
        print(f"   - {nome}: {amostra[i]:.1f} cm")

    # Simular o caminho de decisão
    print(f"\n   🔍 Processo de Decisão:")
    print(
        f"   1. A árvore pergunta: '{iris.feature_names[feature[0]]}' <= {threshold[0]:.2f}?"
    )

    valor_decisao = amostra[feature[0]]
    if valor_decisao <= threshold[0]:
        print(f"   2. Resposta: SIM ({valor_decisao:.2f} <= {threshold[0]:.2f})")
        print(f"   3. Segue para FOLHA ESQUERDA")
    else:
        print(f"   2. Resposta: NÃO ({valor_decisao:.2f} > {threshold[0]:.2f})")
        print(f"   3. Segue para FOLHA DIREITA")

    print(f"   ✅ Classificação final: {classe_pred}")
    print(f"   {'✓ CORRETO' if classe_real == classe_pred else '✗ INCORRETO'}")

# Explicação dos valores nas folhas
print("\n" + "=" * 70)
print("SIGNIFICADO DETALHADO DOS VALORES NAS FOLHAS")
print("=" * 70)

# Informações sobre as folhas
n_leaves = model.get_n_leaves()
children_left = tree.children_left
children_right = tree.children_right
values = tree.value

print(f"\n🍃 A árvore possui {n_leaves} folhas")

folha_num = 1
for i in range(n_nodes):
    # Verificar se é folha (sem filhos)
    if children_left[i] == children_right[i]:  # É uma folha
        samples_node = tree.n_node_samples[i]
        impurity = tree.impurity[i]
        value_counts = values[i][0]

        print(f"\n📍 FOLHA {folha_num}:")
        print(f"   • samples = {samples_node}")
        print(
            f"     └─ Significado: {samples_node} amostras de treinamento chegaram aqui"
        )

        print(f"\n   • value = {value_counts.astype(int).tolist()}")
        print(f"     └─ Distribuição das classes:")
        for j, classe in enumerate(iris.target_names):
            count = int(value_counts[j])
            percent = (count / samples_node * 100) if samples_node > 0 else 0
            print(f"        {classe}: {count} amostras ({percent:.1f}%)")

        print(f"\n   • gini = {impurity:.4f}")
        print(f"     └─ Índice de impureza de Gini")
        if impurity == 0:
            print(f"        Gini = 0 → Folha PURA (100% de uma única classe)")
        else:
            print(f"        Gini > 0 → Folha MISTA (contém múltiplas classes)")
            # Calcular gini manualmente para demonstrar
            probs = value_counts / samples_node
            gini_calc = 1 - sum(probs**2)
            print(
                f"        Cálculo: 1 - Σ(pi²) = 1 - ({' + '.join([f'{p:.3f}²' for p in probs])})"
            )
            print(f"                              = {gini_calc:.4f}")

        classe_majoritaria = iris.target_names[value_counts.argmax()]
        print(f"\n   • class = {classe_majoritaria}")
        print(f"     └─ Classe prevista (maioria dos votos)")

        folha_num += 1

print("\n" + "=" * 70)
print("💡 RESUMO DAS EXPLICAÇÕES")
print("=" * 70)
print(
    """
1. PROCESSO DECISÓRIO:
   A árvore toma decisões sequenciais baseadas em perguntas simples
   do tipo "feature X <= valor?". Cada resposta SIM/NÃO leva a um
   caminho diferente até chegar em uma folha com a classificação final.

2. VALORES NAS FOLHAS (o que significam):
   
   📊 samples: Quantas amostras do conjunto de treino chegaram nesta folha
   
   📈 value: Array com contagem de cada classe [setosa, versicolor, virginica]
             Mostra a distribuição das classes que caíram nesta folha
   
   🎯 gini: Medida de impureza (0 = puro, 1 = máxima mistura)
            Quanto mais próximo de 0, mais confiável é a classificação
            
   🏷️  class: Classe final prevista (a que tem maior contagem no value)

3. EXEMPLO DE INTERPRETAÇÃO:
   Se uma folha tem: samples=50, value=[50, 0, 0], gini=0.0, class=setosa
   
   Significa: 50 amostras chegaram aqui, todas eram setosa (50/0/0),
             a folha é pura (gini=0), então classifica como setosa.
"""
)
