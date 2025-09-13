"""Script principal para execução da Busca Gulosa com visualização."""

import os
import networkx as nx
import numpy as np
from lab.busca.heuristicas import manhattan
from lab.busca.visualizacao_grafo import animar_busca

os.environ["DISPLAY"] = ":1"


if __name__ == "__main__":
    # Criar grafo aleatório maior com ~30 nós e ~35 arestas
    NOS = 30
    grafo = nx.erdos_renyi_graph(NOS, p=0.08)  # p ajustado para ~35 arestas

    # Garantir que o grafo seja conectado
    if not nx.is_connected(grafo):
        # Criar um grafo conectado
        grafo = nx.connected_watts_strogatz_graph(NOS, 4, 0.3)

    # Randomizar start e goal distintos
    rng = np.random.default_rng()
    start = rng.integers(0, NOS)
    goal = rng.integers(0, NOS)
    while start == goal:
        goal = rng.integers(0, NOS)

# Layout orgânico
pos = nx.spring_layout(grafo, k=1.5, iterations=30, seed=42)

try:
    animar_busca(
        grafo,
        start,
        goal,
        pos=pos,
        pausa_passo=0.3,
        pausa_final=1.0,
        heuristica=manhattan,
    )
except (ValueError, TypeError, ImportError, RuntimeError) as e:
    print(f"Erro durante a execução: {e}")
except KeyboardInterrupt:
    print("Execução interrompida pelo usuário")
