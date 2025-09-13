"""Módulo para implementação da Busca Gulosa (Greedy Best-First Search)."""

import heapq
from .heuristicas import manhattan


def busca_gulosa(grafo, start, goal, heuristica=manhattan):
    """
    Executa a busca gulosa em um grafo, usando uma heurística para priorizar nós.

    Args:
        grafo (nx.Graph): Grafo para busca.
        start (tuple): Nó inicial.
        goal (tuple): Nó objetivo.
        heuristica (function): Função heurística (padrão: Manhattan).

    Returns:
        tuple: (visitados: set, parent: dict, encontrado: bool)
            - visitados: Conjunto de nós visitados.
            - parent: Dicionário de pais para reconstruir caminho.
            - encontrado: Se o goal foi alcançado.
    """
    visitados = set()
    fronteira_set = set()
    fronteira = []
    parent = {}

    heapq.heappush(fronteira, (heuristica(start, goal), start))
    fronteira_set.add(start)
    parent[start] = None

    encontrado = False
    while fronteira:
        _, atual = heapq.heappop(fronteira)
        fronteira_set.discard(atual)

        if atual in visitados:
            continue

        visitados.add(atual)

        if atual == goal:
            encontrado = True
            break

        for vizinho in grafo.neighbors(atual):
            if vizinho not in visitados and vizinho not in fronteira_set:
                heapq.heappush(fronteira, (heuristica(vizinho, goal), vizinho))
                fronteira_set.add(vizinho)
                parent[vizinho] = atual

    return visitados, parent, encontrado


def reconstruir_caminho(parent, goal):
    """
    Reconstrói o caminho do início ao goal usando o dicionário parent.

    Args:
        parent (dict): Dicionário de pais.
        goal (tuple): Nó objetivo.

    Returns:
        list: Lista de nós no caminho (início a goal).
    """
    if goal not in parent:
        return []
    caminho_final = []
    temp = goal
    while temp is not None:
        caminho_final.append(temp)
        temp = parent.get(temp)
    caminho_final.reverse()
    return caminho_final
