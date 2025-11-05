"""
Módulo para criar árvores de decisão para operações lógicas.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


class No:
    """Representa um nó na árvore de decisão."""

    def __init__(self, atributo=None, valor=None, esquerda=None, direita=None):
        """
        Inicializa um nó da árvore.

        Args:
            atributo: Nome do atributo testado (ex: 'a', 'b', 'c')
            valor: Valor de classificação (True/False) se for nó folha
            esquerda: Subárvore para quando o teste é False
            direita: Subárvore para quando o teste é True
        """
        self.atributo = atributo
        self.valor = valor
        self.esquerda = esquerda
        self.direita = direita

    def eh_folha(self):
        """Verifica se o nó é uma folha (nó terminal)."""
        return self.valor is not None

    def __str__(self, nivel=0):
        """Representação em string da árvore."""
        indent = "  " * nivel
        if self.eh_folha():
            return f"{indent}→ {self.valor}\n"
        else:
            resultado = f"{indent}{self.atributo}?\n"
            resultado += f"{indent}├─ False:\n{self.esquerda.__str__(nivel + 1)}"
            resultado += f"{indent}└─ True:\n{self.direita.__str__(nivel + 1)}"
            return resultado


def criar_arvore_and(a_label="a", b_label="b"):
    """
    Cria uma árvore de decisão para a operação lógica: a AND b

    Tabela verdade:
    a | b | a AND b
    --|---|--------
    F | F |   F
    F | T |   F
    T | F |   F
    T | T |   T

    Args:
        a_label: Nome do primeiro atributo
        b_label: Nome do segundo atributo

    Returns:
        No raiz da árvore de decisão
    """
    # Se a é False, resultado é False (independente de b)
    # Se a é True, o resultado depende de b
    return No(
        atributo=a_label,
        esquerda=No(valor=False),  # a=False → False
        direita=No(  # a=True → verifica b
            atributo=b_label,
            esquerda=No(valor=False),  # b=False → False
            direita=No(valor=True),  # b=True → True
        ),
    )


def criar_arvore_xor(a_label="a", b_label="b"):
    """
    Cria uma árvore de decisão para a operação lógica: a XOR b

    Tabela verdade:
    a | b | a XOR b
    --|---|--------
    F | F |   F
    F | T |   T
    T | F |   T
    T | T |   F

    Args:
        a_label: Nome do primeiro atributo
        b_label: Nome do segundo atributo

    Returns:
        No raiz da árvore de decisão
    """
    # Se a é False, resultado é b
    # Se a é True, resultado é NOT b
    return No(
        atributo=a_label,
        esquerda=No(  # a=False → depende de b
            atributo=b_label,
            esquerda=No(valor=False),  # b=False → False
            direita=No(valor=True),  # b=True → True
        ),
        direita=No(  # a=True → depende de NOT b
            atributo=b_label,
            esquerda=No(valor=True),  # b=False → True
            direita=No(valor=False),  # b=True → False
        ),
    )


def criar_arvore_complexa():
    """
    Cria uma árvore de decisão para: (a AND b) OR (b AND c)

    Estratégia: Criar DUAS árvores AND completamente separadas
    1. Árvore 1: (a AND b) - com seus próprios nós a e b
    2. Árvore 2: (b AND c) - com seus próprios nós b e c
    3. Combinar com OR: se árvore1=False, vai para árvore2; se árvore1=True, retorna True

    Tabela verdade:
    a | b | c | a AND b | b AND c | resultado
    --|---|---|---------|---------|----------
    F | F | F |    F    |    F    |    F
    F | F | T |    F    |    F    |    F
    F | T | F |    F    |    F    |    F
    F | T | T |    F    |    T    |    T
    T | F | F |    F    |    F    |    F
    T | F | T |    F    |    F    |    F
    T | T | F |    T    |    F    |    T
    T | T | T |    T    |    T    |    T

    Returns:
        No raiz da árvore de decisão
    """

    # ÁRVORE 1: (a AND b) - completamente independente
    arvore_a_and_b = No(
        atributo="a",
        esquerda=No(valor=False),  # a=False → (a AND b)=False
        direita=No(  # a=True → testa b
            atributo="b",
            esquerda=No(valor=False),  # b=False → (a AND b)=False
            direita=No(valor=True),  # b=True → (a AND b)=True
        ),
    )

    # ÁRVORE 2: (b AND c) - completamente independente
    arvore_b_and_c = No(
        atributo="b",
        esquerda=No(valor=False),  # b=False → (b AND c)=False
        direita=No(  # b=True → testa c
            atributo="c",
            esquerda=No(valor=False),  # c=False → (b AND c)=False
            direita=No(valor=True),  # c=True → (b AND c)=True
        ),
    )

    # Agora combinamos com OR de forma explícita:
    # Precisamos testar todos os atributos novamente para simular o OR
    # OR lógico: True se pelo menos uma das árvores retorna True

    # Vamos criar a estrutura que testa tudo:
    # Para cada combinação de a, b, c, calculamos (a AND b) OR (b AND c)

    return No(
        atributo="a",
        esquerda=No(  # a=False → (a AND b)=False, então resultado depende de (b AND c)
            atributo="b",
            esquerda=No(valor=False),  # b=False → (b AND c)=False → resultado=False
            direita=No(  # b=True → depende de c
                atributo="c",
                esquerda=No(valor=False),  # c=False → (b AND c)=False → resultado=False
                direita=No(valor=True),  # c=True → (b AND c)=True → resultado=True
            ),
        ),
        direita=No(  # a=True → depende de b
            atributo="b",
            esquerda=No(  # b=False → (a AND b)=False, então depende de (b AND c)
                atributo="c",
                esquerda=No(valor=False),  # c=False → (b AND c)=False → resultado=False
                direita=No(
                    valor=False
                ),  # c=True → (b AND c)=False (pois b=False) → resultado=False
            ),
            direita=No(
                valor=True
            ),  # b=True → (a AND b)=True → resultado=True (OR curto-circuita)
        ),
    )


def avaliar(arvore, dados):
    """
    Avalia um conjunto de dados usando a árvore de decisão.

    Args:
        arvore: Raiz da árvore de decisão
        dados: Dicionário com valores dos atributos (ex: {'a': True, 'b': False})

    Returns:
        Valor de classificação (True/False)
    """
    if arvore.eh_folha():
        return arvore.valor

    # Navega pela árvore baseado no valor do atributo
    valor_atributo = dados.get(arvore.atributo, False)
    if valor_atributo:
        return avaliar(arvore.direita, dados)
    else:
        return avaliar(arvore.esquerda, dados)


def testar_arvore(arvore, nome_operacao):
    """
    Testa uma árvore de decisão com todas as combinações possíveis.

    Args:
        arvore: Raiz da árvore de decisão
        nome_operacao: Nome da operação para exibição
    """
    print(f"\n{'='*60}")
    print(f"Árvore de decisão para: {nome_operacao}")
    print(f"{'='*60}")
    print("\nEstrutura da árvore:")
    print(arvore)

    # Determina quais atributos são usados
    atributos = set()

    def coletar_atributos(no):
        if not no.eh_folha():
            atributos.add(no.atributo)
            coletar_atributos(no.esquerda)
            coletar_atributos(no.direita)

    coletar_atributos(arvore)
    atributos = sorted(atributos)

    # Testa todas as combinações
    print("\nTeste com todas as combinações:")
    print("-" * 60)

    if len(atributos) == 2:
        header = f"{atributos[0]:^5} | {atributos[1]:^5} | Resultado"
        print(header)
        print("-" * len(header))

        for val_a in [False, True]:
            for val_b in [False, True]:
                dados = {atributos[0]: val_a, atributos[1]: val_b}
                resultado = avaliar(arvore, dados)
                print(f"{str(val_a):^5} | {str(val_b):^5} | {resultado}")

    elif len(atributos) == 3:
        header = (
            f"{atributos[0]:^5} | {atributos[1]:^5} | {atributos[2]:^5} | Resultado"
        )
        print(header)
        print("-" * len(header))

        for val_a in [False, True]:
            for val_b in [False, True]:
                for val_c in [False, True]:
                    dados = {
                        atributos[0]: val_a,
                        atributos[1]: val_b,
                        atributos[2]: val_c,
                    }
                    resultado = avaliar(arvore, dados)
                    print(
                        f"{str(val_a):^5} | {str(val_b):^5} | {str(val_c):^5} | {resultado}"
                    )


def desenhar_arvore(arvore, nome_operacao, nome_arquivo=None):
    """
    Desenha uma árvore de decisão usando matplotlib.

    Args:
        arvore: Raiz da árvore de decisão
        nome_operacao: Nome da operação para o título
        nome_arquivo: Nome do arquivo para salvar (opcional)
    """

    # Conta a profundidade da árvore para ajustar o tamanho
    def profundidade(no):
        if no.eh_folha():
            return 1
        return 1 + max(profundidade(no.esquerda), profundidade(no.direita))

    prof = profundidade(arvore)
    altura = max(10, prof * 2.5)  # Ajusta altura baseado na profundidade
    largura = max(14, prof * 3)  # Ajusta largura baseado na profundidade

    fig, ax = plt.subplots(figsize=(largura, altura))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Título
    ax.text(
        5,
        9.5,
        f"Árvore de Decisão: {nome_operacao}",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    # Contador para posição dos nós
    posicoes = {}

    def calcular_posicoes(no, x, y, largura, id_no=[0], nivel=0):
        """Calcula as posições de todos os nós."""
        no_id = id_no[0]
        id_no[0] += 1
        posicoes[no_id] = {"no": no, "x": x, "y": y}

        if not no.eh_folha():
            # Posiciona filhos com espaçamento adaptativo
            nova_largura = largura / 1.8  # Menos compressão horizontal
            y_filho = y - 1.8  # Mais espaço vertical

            # Filho esquerdo (False)
            id_esq = calcular_posicoes(
                no.esquerda, x - largura / 2, y_filho, nova_largura, id_no, nivel + 1
            )
            # Filho direito (True)
            id_dir = calcular_posicoes(
                no.direita, x + largura / 2, y_filho, nova_largura, id_no, nivel + 1
            )

            posicoes[no_id]["esq"] = id_esq
            posicoes[no_id]["dir"] = id_dir

        return no_id

    # Calcula posições começando da raiz com mais largura inicial
    calcular_posicoes(arvore, 5, 8.5, 3.5)

    # Desenha as arestas primeiro (para ficarem atrás dos nós)
    for no_id, info in posicoes.items():
        no = info["no"]
        x, y = info["x"], info["y"]

        if not no.eh_folha():
            # Aresta esquerda (False)
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
            # Label False
            mid_x = (x + x_esq) / 2 - 0.2
            mid_y = (y + y_esq) / 2
            ax.text(
                mid_x,
                mid_y,
                "False",
                fontsize=9,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"),
            )

            # Aresta direita (True)
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
            # Label True
            mid_x = (x + x_dir) / 2 + 0.2
            mid_y = (y + y_dir) / 2
            ax.text(
                mid_x,
                mid_y,
                "True",
                fontsize=9,
                color="green",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", edgecolor="green"
                ),
            )

    # Desenha os nós
    for no_id, info in posicoes.items():
        no = info["no"]
        x, y = info["x"], info["y"]

        if no.eh_folha():
            # Nó folha (resultado)
            cor = "lightgreen" if no.valor else "lightcoral"
            borda_cor = "darkgreen" if no.valor else "darkred"

            box = FancyBboxPatch(
                (x - 0.4, y - 0.25),
                0.8,
                0.5,
                boxstyle="round,pad=0.1",
                facecolor=cor,
                edgecolor=borda_cor,
                linewidth=2.5,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                str(no.valor),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
        else:
            # Nó de decisão
            circle = plt.Circle(
                (x, y), 0.35, color="lightblue", ec="darkblue", linewidth=2.5, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                f"{no.atributo}?",
                ha="center",
                va="center",
                fontsize=13,
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
            label="Nó de Teste",
        ),
        mpatches.Rectangle(
            (0, 0),
            0.2,
            0.1,
            facecolor="lightgreen",
            edgecolor="darkgreen",
            linewidth=2,
            label="Resultado: True",
        ),
        mpatches.Rectangle(
            (0, 0),
            0.2,
            0.1,
            facecolor="lightcoral",
            edgecolor="darkred",
            linewidth=2,
            label="Resultado: False",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout()

    if nome_arquivo:
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
        print(f"Imagem salva em: {nome_arquivo}")

    plt.show()


if __name__ == "__main__":
    # Testa as três árvores de decisão

    # 1. a AND b
    arvore_and = criar_arvore_and()
    testar_arvore(arvore_and, "a AND b")
    desenhar_arvore(arvore_and, "a AND b", "arvore_and.png")

    # 2. a XOR b
    arvore_xor = criar_arvore_xor()
    testar_arvore(arvore_xor, "a XOR b")
    desenhar_arvore(arvore_xor, "a XOR b", "arvore_xor.png")

    # 3. (a AND b) OR (b AND c)
    arvore_complexa = criar_arvore_complexa()
    testar_arvore(arvore_complexa, "(a AND b) OR (b AND c)")
    desenhar_arvore(arvore_complexa, "(a AND b) OR (b AND c)", "arvore_complexa.png")
