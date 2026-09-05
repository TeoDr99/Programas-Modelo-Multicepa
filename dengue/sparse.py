"""Ruta de acoplamiento disperso (opt-in), para escalar a ``n`` grande.

``build_network`` (en ``dengue.network``) arma ``lam`` y ``delta`` de forma
``(n, c, n)`` densas con solo ~5 vecinos no nulos por nodo (hallazgo 3 de
``CLAUDE.md``): para ``n = 2500`` son cientos de MB y cada evaluación del lado
derecho de :func:`dengue.model.Modelo` cuesta ``O(n²·c)`` en vez de ``O(n·c)``.

Esta ruta es una **alternativa**, no un reemplazo: ``Modelo`` (denso, el que
genera el golden de regresión) no se toca. Se activa a mano (``--sparse`` en
``run.py`` o llamando directamente a lo de acá).

**Supuesto del que depende esta ruta, y que no es una garantía matemática del
modelo:** en ``build_network``, cada asignación (`lam[u, :, u] = beta_h`,
`lam[vecino, :, u] = beta_h * coupling`) se hace por slice sobre **todo** el
eje de cepas a la vez, así que ``lam[:, i, :]`` es la misma matriz ``(n, n)``
para cualquier cepa ``i`` (ídem ``delta``). En general ``lam``/``delta`` son
``(n, c, n)`` y podrían depender de la cepa (ver ``CLAUDE.md`` §4.3 y §4.6); si
en el futuro se necesita acoplamiento dependiente de cepa, esta ruta deja de
ser válida y hay que volver a la densa.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .model import Dimensiones, Mapeo, MapeoInv
from .network import BETA_H_DEFAULT, BETA_V_DEFAULT, COUPLING_DEFAULT, get_idx


def build_network_disperso(grid_rows: int, grid_cols: int, c: int,
                           beta_h: float = BETA_H_DEFAULT, beta_v: float = BETA_V_DEFAULT,
                           coupling: float = COUPLING_DEFAULT
                           ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Construye ``lam`` y ``delta`` como ``scipy.sparse.csr_matrix`` de forma ``(n, n)``.

    Una sola matriz por array (no una por cepa), porque el acoplamiento de
    ``build_network`` no depende de la cepa (ver docstring del módulo). Arma
    las coordenadas directamente con listas, sin pasar por un array denso
    intermedio, para que escale a ``n`` grande.
    """
    n = grid_rows * grid_cols
    filas: list[int] = []
    cols: list[int] = []
    vals_lam: list[float] = []
    vals_delta: list[float] = []

    for r in range(grid_rows):
        for col in range(grid_cols):
            u = get_idx(r, col, grid_cols)
            filas.append(u); cols.append(u)
            vals_lam.append(beta_h); vals_delta.append(beta_v)

            neighbors = []
            if r > 0: neighbors.append((r-1, col))
            if r < grid_rows-1: neighbors.append((r+1, col))
            if col > 0: neighbors.append((r, col-1))
            if col < grid_cols-1: neighbors.append((r, col+1))

            for nr, nc in neighbors:
                v_vec = get_idx(nr, nc, grid_cols)
                filas.append(v_vec); cols.append(u)
                vals_lam.append(beta_h * coupling)
                vals_delta.append(beta_v * coupling)

    lam = sp.csr_matrix((vals_lam, (filas, cols)), shape=(n, n))
    delta = sp.csr_matrix((vals_delta, (filas, cols)), shape=(n, n))
    return lam, delta


def ModeloDisperso(t: float, x: np.ndarray, lam: sp.csr_matrix, mu: float, gamma: float,
                   sigma: np.ndarray, delta: sp.csr_matrix, v: float, N: np.ndarray,
                   dims: Dimensiones) -> np.ndarray:
    """Igual que :func:`dengue.model.Modelo`, con ``lam``/``delta`` dispersas ``(n, n)``.

    ``TasaCont`` y ``dVdt`` son las dos únicas contracciones que involucran a
    ``lam``/``delta``; ambas se reemplazan por un producto disperso-denso
    (``O(nnz·c)`` en vez de ``O(n²·c)``). El resto del cuerpo (``TasaCont2``,
    ``Incidencia``, ``mask``) es ``O(n·c²)`` y queda igual que en ``Modelo``.

    No es bit-idéntico a ``Modelo``: sumar en otro orden (disperso vs. denso)
    cambia los últimos bits del resultado. La comparación en
    ``tests/test_sparse.py`` es a ``1e-10``, no exacta.
    """
    c = dims.c
    S, I, Z, Y, V = MapeoInv(x, dims)
    TasaCont = lam @ V
    mask = np.ones((c, c)) - np.eye(c)
    TasaCont2 = np.einsum('li,lj,ij->lij', Z, TasaCont, sigma * mask)
    Incidencia = S[:, None] * TasaCont

    dSdt = mu * N - np.sum(Incidencia, axis=1) - mu * S
    dIdt = Incidencia - (gamma + mu) * I
    dZdt = gamma * I - np.sum(TasaCont2, axis=2) - mu * Z
    dYdt = TasaCont2 - Y * (mu + gamma)

    Sv = 1 - np.sum(V, axis=1)
    H = I + np.sum(Y, axis=1)
    dVdt = Sv[:, None] * (delta @ H) - v * V

    return Mapeo(dSdt, dIdt, dZdt, dYdt, dVdt)
