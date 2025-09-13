"""Módulo para visualização e animação de buscas em grafos com Matplotlib e NetworkX.

Nota: Para erro 'Authorization required' em Linux, rode: export DISPLAY=:0 ou instale xvfb e execute com 'xvfb-run poetry run python algbusca/gulosa.py'.
"""

import heapq
import networkx as nx
import matplotlib.pyplot as plt
from .busca_gulosa import busca_gulosa, reconstruir_caminho
from .heuristicas import manhattan


def desenhar_grafo(
    grafo,
    pos,
    visitados,
    fronteira_set,
    atual=None,
    path_edges=None,
    encontrado=False,
    start=None,
    goal=None,
    titulo="Busca em Grafo",
):
    """
    Desenha o grafo no estado atual com cores e destaques.

    Args:
        grafo (nx.Graph): Grafo.
        pos (dict): Posições dos nós.
        visitados (set): Nós visitados.
        fronteira_set (set): Nós na fronteira.
        atual (tuple, optional): Nó atual sendo expandido.
        path_edges (list, optional): Arestas do caminho final.
        encontrado (bool): Se o goal foi encontrado.
        start (tuple, optional): Nó inicial.
        goal (tuple, optional): Nó objetivo.
        titulo (str): Título do gráfico.
    """
    plt.clf()
    node_colors = ["lightgray"] * len(grafo.nodes())
    node_sizes = [350] * len(grafo.nodes())

    for idx, node in enumerate(grafo.nodes()):
        if goal and node == goal:
            node_colors[idx] = "green"
            node_sizes[idx] = 450
        elif start and node == start and not encontrado:
            node_colors[idx] = "red"
            node_sizes[idx] = 450
        elif atual and node == atual:
            node_colors[idx] = "orange"
            node_sizes[idx] = 450
        elif node in visitados:
            node_colors[idx] = "blue"
        elif node in fronteira_set:
            node_colors[idx] = "lightgreen"

    nx.draw_networkx_nodes(grafo, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(grafo, pos, alpha=0.3, edge_color="gray", width=0.5)

    if path_edges:
        nx.draw_networkx_edges(
            grafo, pos, edgelist=path_edges, edge_color="darkorange", width=4
        )

    # Labels apenas para início ('I') e alvo ('A')
    if start and goal:
        labels = {start: "I", goal: "A"}
        nx.draw_networkx_labels(grafo, pos, labels, font_size=12, font_weight="bold")

    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()


def animar_busca(
    grafo, start, goal, pos=None, pausa_passo=0.2, pausa_final=1.0, heuristica=None
):
    """
    Anima a busca passo a passo com pausas em tempo real.

    Args:
        grafo (nx.Graph): Grafo.
        start (tuple): Início.
        goal (tuple): Objetivo.
        pos (dict, optional): Posições; se None, usa spring_layout para force-directed.
        pausa_passo (float): Pausa entre passos.
        pausa_final (float): Pausa no final.
        heuristica (function, optional): Heurística para usar na busca.
    """
    if heuristica is None:
        heuristica = manhattan

    # Gerar layout force-directed se não fornecido (simplificado para menos caos)
    if pos is None:
        pos = nx.spring_layout(grafo, k=1.5, iterations=30, seed=42)

    _, parent, encontrado = busca_gulosa(grafo, start, goal, heuristica)

    if not encontrado:
        print(f"Busca falhou. Início em {start}, Alvo em {goal}")
        fig = plt.figure(figsize=(20, 20))
        desenhar_grafo(
            grafo, pos, set(), set(), encontrado=False, start=start, goal=goal
        )
        plt.show()
        return

    # Reconstruir caminho
    caminho_final = reconstruir_caminho(parent, goal)
    path_edges = (
        list(zip(caminho_final[:-1], caminho_final[1:]))
        if len(caminho_final) > 1
        else []
    )

    # Execução com desenho em cada passo
    fig = plt.figure(figsize=(20, 20))
    plt.ion()
    visitados_sim = set()
    fronteira_set_sim = set()
    fronteira_sim = []
    parent_sim = {}
    heapq.heappush(fronteira_sim, (heuristica(start, goal), start))
    fronteira_set_sim.add(start)
    parent_sim[start] = None

    atual = None
    passo = 0
    while fronteira_sim:
        _, atual = heapq.heappop(fronteira_sim)
        fronteira_set_sim.discard(atual)

        if atual in visitados_sim:
            continue

        visitados_sim.add(atual)
        titulo_passo = (
            f"Busca Gulosa - Passo {passo + 1} em Andamento (Heurística: Manhattan)"
        )
        desenhar_grafo(
            grafo,
            pos,
            visitados_sim,
            fronteira_set_sim,
            atual,
            start=start,
            goal=goal,
            titulo=titulo_passo,
        )
        plt.pause(pausa_passo)
        passo += 1

        if atual == goal:
            desenhar_grafo(
                grafo,
                pos,
                visitados_sim,
                fronteira_set_sim,
                None,
                path_edges,
                True,
                start,
                goal,
                titulo="Busca Gulosa - Encontrado!",
            )
            plt.pause(pausa_final)
            break

        for vizinho in grafo.neighbors(atual):
            if vizinho not in visitados_sim and vizinho not in fronteira_set_sim:
                heapq.heappush(fronteira_sim, (heuristica(vizinho, goal), vizinho))
                fronteira_set_sim.add(vizinho)
                parent_sim[vizinho] = atual

    plt.ioff()
    plt.show()
