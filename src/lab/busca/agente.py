""" "Módulo que define a classe Agente, representando um agente em um ambiente de busca."""

import turtle


class Agente:
    """Classe que representa um agente em um ambiente de busca."""

    def __init__(self, grade, linha, coluna, cor="black", forma="turtle"):
        self.direcoes_possiveis = {
            "norte": (1, 0),
            "sul": (-1, 0),
            "oeste": (0, 1),
            "leste": (0, -1),
        }
        self.grade = grade
        self.linha = linha
        self.coluna = coluna
        self.turtle = turtle.Turtle(shape=forma)
        self.turtle.color(cor)
        self.turtle.penup()

    def move(self, linha, coluna):
        """ "Move o agente para a posição (linha, coluna)."""
        self.linha = linha
        self.coluna = coluna
        x, y = self.grade(self.linha, self.coluna)
        self.turtle.goto(x, y)

    @property
    def posicao(self):
        """Retorna a posição atual do agente como uma tupla (linha, coluna)."""
        return self.linha, self.coluna

    @property
    def sucessores(self):
        """Retorna uma lista de posições (linha, coluna) que o agente pode alcançar a partir da sua posição atual."""
        lst = []
        for _, (l, c) in self.direcoes_possiveis.items():
            linha = self.linha + l
            coluna = self.coluna + c
            if 1 <= linha <= self.grade.nlinhas and 1 <= coluna <= self.grade.ncolunas:
                lst.append((linha, coluna))
        return lst

    def __repr__(self):
        return f"Agente({self.linha}, {self.coluna})"

    def __eq__(self, other):
        return (self.linha, self.coluna) == (other.linha, other.coluna)
