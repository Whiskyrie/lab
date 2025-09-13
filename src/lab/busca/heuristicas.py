"""Módulo com heurísticas comuns para buscas informadas."""

import numpy as np


def manhattan(posicao, objetivo):
    """
    Calcula a distância de Manhattan entre posicao e objetivo.

    Args:
        posicao: Posição atual (tuple para grade ou int/np.integer para grafo numerado).
        objetivo: Posição do objetivo (tuple ou int/np.integer).

    Returns:
        int/float: Distância (Manhattan para tuples, absoluta para ints).
    """
    if isinstance(posicao, (int, np.integer)) and isinstance(
        objetivo, (int, np.integer)
    ):
        return abs(
            int(posicao) - int(objetivo)
        )  # Para grafos numerados simples (ex.: nós 1-5 ou np.int64)
    else:
        linha, coluna = posicao
        alvo_linha, alvo_coluna = objetivo
        return abs(linha - alvo_linha) + abs(coluna - alvo_coluna)


def euclidiana(posicao, objetivo):
    """
    Calcula a distância euclidiana entre posicao e objetivo.

    Args:
        posicao (tuple): Posição atual (linha, coluna).
        objetivo (tuple): Posição do objetivo (linha, coluna).

    Returns:
        float: Distância euclidiana.
    """
    linha, coluna = posicao
    alvo_linha, alvo_coluna = objetivo
    return np.sqrt((linha - alvo_linha) ** 2 + (coluna - alvo_coluna) ** 2)
