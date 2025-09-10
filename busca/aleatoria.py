"""Busca em largura com expansão aleatória dos nós"""

import os
import turtle
from collections import deque
import numpy as np
from lab.busca import sorteia_coords, embaralha
from lab.busca.agente import Agente
from lab.busca.alvo import Alvo
from lab.busca.grade import Grade


os.environ["DISPLAY"] = ":1"

rnd = np.random.default_rng(5)
grade = Grade(fps=5)
agente = Agente(grade, linha=10, coluna=10)
alvo = Alvo(grade, *sorteia_coords(grade, rnd))
visitados = set()
sucessores = deque([agente.posicao])
while agente != alvo and sucessores:
    embaralha(sucessores, rnd)
    proximo = sucessores.pop()
    sucessores.clear()
    agente.move(*proximo)
    visitados.add(proximo)

    # Verifica se chegou ao alvo
    if agente == alvo:
        grade.pinta(
            *agente.posicao, cor="green"
        )  # Marca em verde quando encontra o alvo
        break

    for sucessor in agente.sucessores:
        if sucessor not in visitados:
            grade.pinta(*sucessor, cor="lightgreen")
            sucessores.append(sucessor)
    grade.pinta(*agente.posicao, cor="blue")
    grade.desenha()

# Verificação final e pintura final
if agente == alvo:
    grade.pinta(*agente.posicao, cor="green")
else:
    print(
        f"Busca falhou. Agente em {agente.posicao}, Alvo em {(alvo.linha, alvo.coluna)}"
    )
    grade.pinta(*agente.posicao, cor="black")

grade.desenha()
turtle.done()
