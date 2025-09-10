"""Busca em largura (BFS - Breadth-First Search)."""

import os
import turtle
from collections import deque
import numpy as np
from lab.busca import sorteia_coords
from lab.busca.agente import Agente
from lab.busca.alvo import Alvo
from lab.busca.grade import Grade

os.environ["DISPLAY"] = ":1"

rnd = np.random.default_rng(23)
grade = Grade(fps=10)
agente = Agente(grade, linha=8, coluna=8)
alvo = Alvo(grade, *sorteia_coords(grade, rnd))
visitados = set()
fronteira = deque([agente.posicao])
"""
Dicas:
???????? = fronteira.popleft()   ← Retira da lista o elemento mais à esquerda (FIFO).
fronteira.append(??????????)    ← Insere elemento à direita da fila.
"""
while agente != alvo and fronteira:
    proximo = fronteira.popleft()

    if proximo in visitados:
        continue

    agente.move(*proximo)
    grade.pinta(*agente.posicao, cor="blue")
    visitados.add(agente.posicao)

    # Verifica se chegou ao alvo
    if agente == alvo:
        grade.pinta(
            *agente.posicao, cor="green"
        )  # Marca em verde quando encontra o alvo
        break

    for vizinho in agente.sucessores:
        if vizinho not in visitados and vizinho not in fronteira:
            grade.pinta(*vizinho, cor="palegreen")
            fronteira.append(vizinho)

    grade.pinta(*agente.posicao, cor="blue")
    grade.desenha()

# Verificação final
if agente == alvo:
    grade.pinta(*agente.posicao, cor="green")
else:
    print(
        f"Busca falhou. Agente em {agente.posicao}, Alvo em {(alvo.linha, alvo.coluna)}"
    )

grade.desenha()

turtle.done()
