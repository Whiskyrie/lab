"""Módulo para criação de grafos de grade genéricos."""

import networkx as nx


def criar_grafo_grade(nlinhas, ncolunas):
    """
    Cria um grafo de grade com nós 1-based e arestas para vizinhos (norte, sul, leste, oeste).

    Args:
        nlinhas (int): Número de linhas na grade.
        ncolunas (int): Número de colunas na grade.

    Returns:
        nx.Graph: Grafo de grade criado.
    """
    grafo = nx.Graph()
    for i in range(1, nlinhas + 1):
        for j in range(1, ncolunas + 1):
            grafo.add_node((i, j))

    for i in range(1, nlinhas + 1):
        for j in range(1, ncolunas + 1):
            node = (i, j)
            # Norte
            if i > 1:
                grafo.add_edge(node, (i - 1, j))
            # Sul
            if i < nlinhas:
                grafo.add_edge(node, (i + 1, j))
            # Oeste
            if j > 1:
                grafo.add_edge(node, (i, j - 1))
            # Leste
            if j < ncolunas:
                grafo.add_edge(node, (i, j + 1))
    return grafo


def posicoes_layout(nlinhas, ncolunas):
    """
    Gera dicionário de posições para layout da grade (inverte eixo y para visualização padrão).

    Args:
        nlinhas (int): Número de linhas na grade.
        ncolunas (int): Número de colunas na grade.

    Returns:
        dict: Dicionário de posições {(linha, coluna): (x, y)}.
    """
    return {
        (i, j): (j - 1, nlinhas - i)
        for i in range(1, nlinhas + 1)
        for j in range(1, ncolunas + 1)
    }
