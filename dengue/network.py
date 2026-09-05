"""Geometría de la grilla y construcción de la red espacial (acoplamiento denso).

Movido verbatim desde ``dengue/model.py`` (Sesión B, B1): sin cambios de
comportamiento. No depende de nada de ``dengue.model``.
"""

from __future__ import annotations

import numpy as np

BETA_H_DEFAULT = 0.5            # transmisión mosquito -> humano (diagonal de lam)
BETA_V_DEFAULT = 0.5            # transmisión humano -> mosquito (diagonal de delta)
COUPLING_DEFAULT = 0.05         # factor de acoplamiento con primeros vecinos
SIGMA_DEFAULT = 1.5             # efecto cruzado fuera de la diagonal (enhancement)


def geometria_grilla(n_deseado: int) -> tuple[int, int, int]:
    """Devuelve ``(grid_rows, grid_cols, n)`` para la grilla más cuadrada posible.

    Reproduce el cálculo del script original: busca el mayor divisor de
    ``n_deseado`` que no supere su raíz cuadrada.
    """
    root = int(np.sqrt(n_deseado))
    while n_deseado % root != 0:
        root -= 1
    grid_rows = root
    grid_cols = n_deseado // root
    n = grid_rows * grid_cols
    return grid_rows, grid_cols, n


def get_idx(r: int, col: int, grid_cols: int) -> int:
    """Índice lineal del nodo en la fila ``r`` y columna ``col``."""
    return r * grid_cols + col


def build_network(grid_rows: int, grid_cols: int, c: int,
                  beta_h: float = BETA_H_DEFAULT, beta_v: float = BETA_V_DEFAULT,
                  coupling: float = COUPLING_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Construye ``lam`` y ``delta`` (forma ``(n, c, n)``) sobre una grilla.

    Cada nodo se acopla consigo mismo con ``beta`` y con sus primeros vecinos
    (arriba, abajo, izquierda, derecha) con ``beta * coupling``. Mismo patrón
    de llenado que el script original: ``lam[vecino, :, u]``.

    Nota: cada asignación se hace por slice sobre **todo** el eje de cepas a
    la vez, así que ``lam[:, i, :]`` (y ``delta[:, i, :]``) es la misma matriz
    ``(n, n)`` para cualquier cepa ``i``. Esta invariancia es la que aprovecha
    la ruta dispersa opt-in de ``dengue.sparse``.
    """
    n = grid_rows * grid_cols
    lam = np.zeros((n, c, n))
    delta = np.zeros((n, c, n))

    for r in range(grid_rows):
        for col in range(grid_cols):
            u = get_idx(r, col, grid_cols)
            lam[u, :, u] = beta_h
            delta[u, :, u] = beta_v

            neighbors = []
            if r > 0: neighbors.append((r-1, col))
            if r < grid_rows-1: neighbors.append((r+1, col))
            if col > 0: neighbors.append((r, col-1))
            if col < grid_cols-1: neighbors.append((r, col+1))

            for nr, nc in neighbors:
                v_vec = get_idx(nr, nc, grid_cols)
                lam[v_vec, :, u] = beta_h * coupling
                delta[v_vec, :, u] = beta_v * coupling

    return lam, delta


def sigma_default(c: int, valor: float = SIGMA_DEFAULT) -> np.ndarray:
    """Matriz ``sigma`` del script original: ``valor`` fuera de la diagonal, 0 en ella."""
    sigma = np.ones((c, c)) * valor
    np.fill_diagonal(sigma, 0.0)
    return sigma
