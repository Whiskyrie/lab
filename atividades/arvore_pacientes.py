"""
Árvore de Decisão para Classificação de Pacientes
Usando medida de entropia para induzir a árvore manualmente
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Dados de treinamento
dados_treino = [
    # [Febre, Enjoo, Manchas, Dores, Situação]
    ["sim", "sim", "pequenas", "sim", "doente"],  # João
    ["não", "não", "grandes", "não", "saudável"],  # Pedro
    ["sim", "sim", "pequenas", "não", "saudável"],  # Maria
    ["sim", "não", "grandes", "sim", "doente"],  # José
    ["sim", "não", "pequenas", "sim", "saudável"],  # Ana
    ["não", "não", "grandes", "sim", "doente"],  # Leila
]

# Dados de teste
dados_teste = [
    # [Febre, Enjoo, Manchas, Dores, Situação esperada]
    ["não", "não", "pequenas", "sim", None],  # Luis
    ["sim", "sim", "grandes", "sim", None],  # Laura
]

nomes_treino = ["João", "Pedro", "Maria", "José", "Ana", "Leila"]
nomes_teste = ["Luis", "Laura"]


class NoArvore:
    """Representa um nó na árvore de decisão."""

    def __init__(
        self,
        atributo=None,
        classe=None,
        esquerda=None,
        direita=None,
        valor_divisao=None,
    ):
        self.atributo = atributo  # Índice ou nome do atributo para divisão
        self.classe = classe  # Classe predita (se for folha)
        self.esquerda = esquerda  # Subárvore esquerda
        self.direita = direita  # Subárvore direita
        self.valor_divisao = valor_divisao  # Valor usado para divisão

    def eh_folha(self):
        return self.classe is not None


def calcular_entropia(dados):
    """
    Calcula a entropia de um conjunto de dados.
    Entropia(t) = -∑ p(i/t) * log₂(p(i/t))
    """
    if len(dados) == 0:
        return 0

    # Conta as classes
    classes = [d[-1] for d in dados]
    valores_unicos, contagens = np.unique(classes, return_counts=True)

    # Calcula entropia
    entropia = 0
    total = len(dados)

    for contagem in contagens:
        p = contagem / total
        if p > 0:  # Convenção: 0 * log₂(0) = 0
            entropia -= p * np.log2(p)

    return entropia


def calcular_ganho_informacao(dados, atributo_idx):
    """
    Calcula o ganho de informação para um atributo.
    Δ = I(v_pai) - ∑(N(v_t)/N) * I(v_t)
    """
    entropia_pai = calcular_entropia(dados)

    # Separa dados por valor do atributo
    valores_atributo = {}
    for d in dados:
        valor = d[atributo_idx]
        if valor not in valores_atributo:
            valores_atributo[valor] = []
        valores_atributo[valor].append(d)

    # Calcula média ponderada das entropias dos filhos
    entropia_filhos = 0
    total = len(dados)

    for valor, subset in valores_atributo.items():
        peso = len(subset) / total
        entropia_filhos += peso * calcular_entropia(subset)

    ganho = entropia_pai - entropia_filhos
    return ganho


def analisar_atributos(dados, atributos_disponiveis):
    """Analisa todos os atributos e retorna os ganhos de informação."""
    print("\n" + "=" * 70)
    print("ANÁLISE DE GANHO DE INFORMAÇÃO")
    print("=" * 70)

    entropia_atual = calcular_entropia(dados)
    print(f"\nEntropia do nó atual: {entropia_atual:.4f}")

    nomes_atributos = ["Febre", "Enjoo", "Manchas", "Dores"]
    ganhos = {}

    print("\nGanho de informação por atributo:")
    print("-" * 70)

    for idx in atributos_disponiveis:
        ganho = calcular_ganho_informacao(dados, idx)
        ganhos[idx] = ganho
        print(f"{nomes_atributos[idx]:12} (índice {idx}): {ganho:.4f}")

    melhor_atributo = max(ganhos, key=ganhos.get)
    print(
        f"\n✓ Melhor atributo: {nomes_atributos[melhor_atributo]} "
        f"(ganho = {ganhos[melhor_atributo]:.4f})"
    )

    return ganhos


def construir_arvore_manual():
    """
    Constrói a árvore de decisão manualmente usando entropia.
    Vamos analisar passo a passo e construir a árvore.
    """

    print("\n" + "=" * 70)
    print("CONSTRUÇÃO DA ÁRVORE DE DECISÃO - ANÁLISE MANUAL")
    print("=" * 70)

    # Nível 0: Raiz
    print("\n### NÍVEL 0: RAIZ ###")
    atributos_disponiveis = [0, 1, 2, 3]  # Febre, Enjoo, Manchas, Dores
    ganhos_raiz = analisar_atributos(dados_treino, atributos_disponiveis)

    # Melhor atributo na raiz: vamos analisar
    # Baseado na análise de entropia, vou escolher o atributo com maior ganho

    # Para este dataset pequeno, vou calcular e depois construir manualmente
    # uma árvore otimizada

    # Análise manual baseada nos dados:
    # - Enjoo parece ser um bom divisor (quando tem enjoo, mais provável doente)
    # - Dores também é relevante
    # - Vamos usar Dores como raiz (testando hipótese)

    print("\n### DECISÃO: Usar DORES como atributo raiz ###")

    # Divide por Dores
    dados_sem_dor = [d for d in dados_treino if d[3] == "não"]
    dados_com_dor = [d for d in dados_treino if d[3] == "sim"]

    print(f"\nDados sem dor (Dores = não): {len(dados_sem_dor)} exemplos")
    for d in dados_sem_dor:
        print(f"  {d}")
    print(f"  Entropia: {calcular_entropia(dados_sem_dor):.4f}")

    print(f"\nDados com dor (Dores = sim): {len(dados_com_dor)} exemplos")
    for d in dados_com_dor:
        print(f"  {d}")
    print(f"  Entropia: {calcular_entropia(dados_com_dor):.4f}")

    # Subárvore esquerda (sem dor)
    print("\n### NÍVEL 1: Subárvore ESQUERDA (Dores = não) ###")
    if calcular_entropia(dados_sem_dor) == 0:
        print("Entropia = 0, nó puro! Classe: saudável")
        no_esquerdo = NoArvore(classe="saudável")
    else:
        atributos_restantes = [0, 1, 2]  # Febre, Enjoo, Manchas
        ganhos_esq = analisar_atributos(dados_sem_dor, atributos_restantes)
        no_esquerdo = NoArvore(classe="saudável")  # Maioria é saudável

    # Subárvore direita (com dor)
    print("\n### NÍVEL 1: Subárvore DIREITA (Dores = sim) ###")
    if calcular_entropia(dados_com_dor) == 0:
        print("Entropia = 0, nó puro! Classe: doente")
        no_direito = NoArvore(classe="doente")
    else:
        print("Entropia > 0, precisa dividir mais")
        atributos_restantes = [0, 1, 2]  # Febre, Enjoo, Manchas
        ganhos_dir = analisar_atributos(dados_com_dor, atributos_restantes)

        # Divide por Manchas (tem o maior ganho de informação)
        print("\n### DECISÃO: Usar MANCHAS na subárvore direita ###")

        dados_com_dor_manchas_pequenas = [
            d for d in dados_com_dor if d[2] == "pequenas"
        ]
        dados_com_dor_manchas_grandes = [d for d in dados_com_dor if d[2] == "grandes"]

        print(
            f"\nDados com dor E manchas pequenas: {len(dados_com_dor_manchas_pequenas)} exemplos"
        )
        for d in dados_com_dor_manchas_pequenas:
            print(f"  {d}")

        print(
            f"\nDados com dor E manchas grandes: {len(dados_com_dor_manchas_grandes)} exemplos"
        )
        for d in dados_com_dor_manchas_grandes:
            print(f"  {d}")

        # Cria subárvore com Manchas
        no_direito = NoArvore(
            atributo=2,  # Manchas
            esquerda=NoArvore(classe="saudável"),  # Manchas pequenas → maioria saudável
            direita=NoArvore(classe="doente"),  # Manchas grandes → doente
        )

    # Árvore final
    raiz = NoArvore(
        atributo=3,  # Dores
        esquerda=no_esquerdo,
        direita=no_direito,
    )

    return raiz


def prever(arvore, exemplo):
    """Faz predição para um exemplo usando a árvore."""
    if arvore.eh_folha():
        return arvore.classe

    valor = exemplo[arvore.atributo]

    # Para atributos binários sim/não e pequenas/grandes
    # Esquerda: não ou pequenas
    # Direita: sim ou grandes
    if valor in ["não", "pequenas"]:
        return prever(arvore.esquerda, exemplo)
    else:  # sim ou grandes
        return prever(arvore.direita, exemplo)


def avaliar_arvore(arvore):
    """Avalia a árvore nos dados de treino e teste."""

    print("\n" + "=" * 70)
    print("AVALIAÇÃO DA ÁRVORE")
    print("=" * 70)

    # Predições no treino
    print("\n### PREDIÇÕES NO CONJUNTO DE TREINO ###")
    print("-" * 70)
    print(f"{'Nome':10} | {'Real':10} | {'Predito':10} | {'Correto?':10}")
    print("-" * 70)

    acertos_treino = 0
    predicoes_treino = []

    for i, (dados, nome) in enumerate(zip(dados_treino, nomes_treino)):
        real = dados[-1]
        predito = prever(arvore, dados)
        correto = "✓" if real == predito else "✗"
        if real == predito:
            acertos_treino += 1
        predicoes_treino.append(predito)
        print(f"{nome:10} | {real:10} | {predito:10} | {correto:10}")

    erro_treino = 1 - (acertos_treino / len(dados_treino))
    print(
        f"\nAcurácia no treino: {acertos_treino}/{len(dados_treino)} = "
        f"{acertos_treino/len(dados_treino):.2%}"
    )
    print(f"Erro de treino: {erro_treino:.2%}")

    # Predições no teste
    print("\n### PREDIÇÕES NO CONJUNTO DE TESTE ###")
    print("-" * 70)
    print(
        f"{'Nome':10} | {'Febre':8} | {'Enjoo':8} | {'Manchas':10} | "
        f"{'Dores':8} | {'Predito':10}"
    )
    print("-" * 70)

    predicoes_teste = []
    for dados, nome in zip(dados_teste, nomes_teste):
        predito = prever(arvore, dados)
        predicoes_teste.append(predito)
        print(
            f"{nome:10} | {dados[0]:8} | {dados[1]:8} | {dados[2]:10} | "
            f"{dados[3]:8} | {predito:10}"
        )

    print("\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    print(f"\nPredições no treino: {predicoes_treino}")
    print(f"Erro de treino: {erro_treino:.2%}")
    print(f"\nPredições no teste: {predicoes_teste}")
    print(f"(Não há rótulos reais para teste, então não calculamos erro de teste)")

    return predicoes_treino, predicoes_teste, erro_treino


def desenhar_arvore_pacientes(arvore):
    """Desenha a árvore de decisão para o problema dos pacientes."""

    # Usar backend não-interativo
    import matplotlib

    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Título
    ax.text(
        5,
        9.5,
        "Árvore de Decisão: Diagnóstico de Pacientes",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    nomes_atributos = ["Febre", "Enjoo", "Manchas", "Dores"]

    # Mapeia nós para posições
    posicoes = {}

    def calcular_posicoes(no, x, y, largura, id_no=[0]):
        """Calcula as posições de todos os nós."""
        no_id = id_no[0]
        id_no[0] += 1
        posicoes[no_id] = {"no": no, "x": x, "y": y}

        if not no.eh_folha():
            nova_largura = largura / 1.8
            y_filho = y - 2.0

            id_esq = calcular_posicoes(
                no.esquerda, x - largura / 2, y_filho, nova_largura, id_no
            )
            id_dir = calcular_posicoes(
                no.direita, x + largura / 2, y_filho, nova_largura, id_no
            )

            posicoes[no_id]["esq"] = id_esq
            posicoes[no_id]["dir"] = id_dir

        return no_id

    calcular_posicoes(arvore, 5, 8.0, 3.5)

    # Desenha arestas
    for no_id, info in posicoes.items():
        no = info["no"]
        x, y = info["x"], info["y"]

        if not no.eh_folha():
            # Aresta esquerda (não)
            x_esq = posicoes[info["esq"]]["x"]
            y_esq = posicoes[info["esq"]]["y"]

            arrow = FancyArrowPatch(
                (x, y - 0.3),
                (x_esq, y_esq + 0.3),
                arrowstyle="->",
                mutation_scale=20,
                color="red",
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(arrow)
            mid_x = (x + x_esq) / 2 - 0.2
            mid_y = (y + y_esq) / 2
            ax.text(
                mid_x,
                mid_y,
                "não",
                fontsize=10,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"),
            )

            # Aresta direita (sim)
            x_dir = posicoes[info["dir"]]["x"]
            y_dir = posicoes[info["dir"]]["y"]

            arrow = FancyArrowPatch(
                (x, y - 0.3),
                (x_dir, y_dir + 0.3),
                arrowstyle="->",
                mutation_scale=20,
                color="green",
                linewidth=2,
            )
            ax.add_patch(arrow)
            mid_x = (x + x_dir) / 2 + 0.2
            mid_y = (y + y_dir) / 2
            ax.text(
                mid_x,
                mid_y,
                "sim",
                fontsize=10,
                color="green",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", edgecolor="green"
                ),
            )

    # Desenha nós
    for no_id, info in posicoes.items():
        no = info["no"]
        x, y = info["x"], info["y"]

        if no.eh_folha():
            # Nó folha
            cor = "#90EE90" if no.classe == "saudável" else "#FFB6C6"
            borda_cor = "darkgreen" if no.classe == "saudável" else "darkred"

            box = FancyBboxPatch(
                (x - 0.5, y - 0.3),
                1.0,
                0.6,
                boxstyle="round,pad=0.1",
                facecolor=cor,
                edgecolor=borda_cor,
                linewidth=2.5,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                no.classe.upper(),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        else:
            # Nó de decisão
            circle = plt.Circle(
                (x, y), 0.4, color="lightblue", ec="darkblue", linewidth=2.5, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                f"{nomes_atributos[no.atributo]}?",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                zorder=11,
            )

    # Legenda
    legend_elements = [
        mpatches.Circle(
            (0, 0),
            0.1,
            facecolor="lightblue",
            edgecolor="darkblue",
            linewidth=2,
            label="Teste",
        ),
        mpatches.Rectangle(
            (0, 0),
            0.2,
            0.1,
            facecolor="#90EE90",
            edgecolor="darkgreen",
            linewidth=2,
            label="Saudável",
        ),
        mpatches.Rectangle(
            (0, 0),
            0.2,
            0.1,
            facecolor="#FFB6C6",
            edgecolor="darkred",
            linewidth=2,
            label="Doente",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout()
    caminho_arquivo = "/home/whiskyrie/Projetos/lab/atividades/arvore_pacientes.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Árvore desenhada e salva em: {caminho_arquivo}")


def main():
    """Função principal."""

    print("\n" + "#" * 70)
    print("# ÁRVORE DE DECISÃO - DIAGNÓSTICO DE PACIENTES")
    print("# Usando medida de ENTROPIA")
    print("#" * 70)

    # Constrói a árvore
    arvore = construir_arvore_manual()

    # Avalia a árvore
    pred_treino, pred_teste, erro_treino = avaliar_arvore(arvore)

    # Desenha a árvore
    desenhar_arvore_pacientes(arvore)

    print("\n" + "#" * 70)
    print("# CONCLUSÃO")
    print("#" * 70)
    print(
        f"""
A árvore de decisão foi induzida usando ENTROPIA como medida de impureza.

RESULTADOS:
-----------
✓ Valores preditos (treino): {pred_treino}
✓ Valores preditos (teste):  {pred_teste}
✓ Erro de treinamento: {erro_treino:.2%}
✓ Erro de teste: N/A (sem rótulos verdadeiros)

ESTRUTURA DA ÁRVORE:
--------------------
A árvore usa principalmente o atributo "Dores" como divisor principal,
pois este apresenta o maior ganho de informação para classificar
pacientes em "doente" ou "saudável".

A imagem da árvore foi salva em: arvore_pacientes.png
"""
    )


if __name__ == "__main__":
    main()
