"""Modelo compartimental multicepa de dengue con estructura espacial.

Extracción **sin cambios semánticos** del script original
``Estudio de modelo epidemológico - C1.py`` (hoy ``run.py``): mismos ``einsum``,
mismos índices, mismos valores por defecto y mismo ``order='F'``.

Este módulo se queda con el núcleo dinámico (``Dimensiones``, ``Mapeo``,
``MapeoInv``, ``Modelo``) y reexporta la construcción de la red espacial
(``dengue.network``) y las condiciones iniciales (``dengue.scenarios``), que
viven en sus propios módulos desde la Sesión B. La fachada existe porque toda
la suite de tests de la Sesión A importa exclusivamente con
``from dengue.model import ...`` y ese contrato es congelado: no se toca.

Este módulo no importa ``matplotlib`` a propósito, para poder importarlo desde
los tests sin abrir ninguna ventana.

Convención de índices (normativa, ver ``CLAUDE.md`` sección 2):

- ``l``, ``m`` en ``{0, ..., n-1}`` recorren parches espaciales.
- ``i``, ``j`` en ``{0, ..., c-1}`` recorren serotipos.
- ``Y[l, i, j]``: infección **primaria** por ``i``, **secundaria** por ``j``.
  Para contar humanos que infectan con la cepa ``j`` se suma sobre el eje 1.
"""

from __future__ import annotations

import numpy as np

from .network import (
    BETA_H_DEFAULT, BETA_V_DEFAULT, COUPLING_DEFAULT, SIGMA_DEFAULT,
    build_network, geometria_grilla, get_idx, sigma_default,
)
from .scenarios import condiciones_iniciales

# ============================================================================
# Parámetros por defecto del script original
# ============================================================================
MU_DEFAULT = 1.0 / (70 * 365)   # natalidad/mortalidad humana (1/días)
GAMMA_DEFAULT = 1.0 / 7.0       # tasa de recuperación humana (1/días)
NU_DEFAULT = 1.0 / 14.0         # mortalidad del mosquito (1/días)


# ============================================================================
# 2. GESTIÓN DE MEMORIA
# ============================================================================
class Dimensiones:
    """Tamaños del sistema y cortes del vector de estado.

    Reemplaza a las variables globales ``n``, ``c`` y ``cuts`` del script
    original. ``cuts`` es el ``cumsum`` de los tamaños de los bloques
    ``S, I, Z, Y, V`` (en ese orden) y lo usa :func:`MapeoInv`.
    """

    def __init__(self, n: int, c: int) -> None:
        self.n = int(n)
        self.c = int(c)
        size_S = self.n
        size_I = self.n * self.c
        size_Z = self.n * self.c
        size_Y = self.n * self.c * self.c
        size_V = self.n * self.c
        self.cuts = np.cumsum([size_S, size_I, size_Z, size_Y, size_V])

    @property
    def total(self) -> int:
        """Longitud del vector de estado aplanado."""
        return int(self.cuts[-1])

    def __repr__(self) -> str:
        return f"Dimensiones(n={self.n}, c={self.c})"


def Mapeo(S: np.ndarray, I: np.ndarray, Z: np.ndarray, Y: np.ndarray,
          V: np.ndarray) -> np.ndarray:
    """Aplana ``(S, I, Z, Y, V)`` en un único vector de estado (orden Fortran)."""
    return np.concatenate([
        S.flatten(order='F'), I.flatten(order='F'),
        Z.flatten(order='F'), Y.flatten(order='F'), V.flatten(order='F')
    ])


def MapeoInv(x: np.ndarray, dims: Dimensiones) -> tuple[np.ndarray, ...]:
    """Inversa de :func:`Mapeo`: devuelve ``(S, I, Z, Y, V)`` con sus formas."""
    n, c, cuts = dims.n, dims.c, dims.cuts
    s, i, z, y, v = np.split(x, cuts[:-1])
    return (s,
            i.reshape((n, c), order='F'),
            z.reshape((n, c), order='F'),
            y.reshape((n, c, c), order='F'),
            v.reshape((n, c), order='F'))


# ============================================================================
# 3. MODELO DINÁMICO
# ============================================================================
def Modelo(t: float, x: np.ndarray, lam: np.ndarray, mu: float, gamma: float,
           sigma: np.ndarray, delta: np.ndarray, v: float, N: np.ndarray,
           dims: Dimensiones) -> np.ndarray:
    """Lado derecho del sistema de EDOs (firma compatible con ``solve_ivp``).

    ``lam`` y ``delta`` tienen forma ``(n, c, n)``, ``sigma`` forma ``(c, c)``,
    ``N`` forma ``(n,)``; ``mu``, ``gamma`` y ``v`` (la mortalidad del mosquito,
    ν en el TEX) son escalares.
    """
    c = dims.c
    S, I, Z, Y, V = MapeoInv(x, dims)
    TasaCont = np.einsum('lim,mi->li', lam, V)
    mask = np.ones((c, c)) - np.eye(c)
    TasaCont2 = np.einsum('li,lj,ij->lij', Z, TasaCont, sigma * mask)
    Incidencia = np.einsum('l,li->li', S, TasaCont)

    dSdt = mu * N - np.sum(Incidencia, axis=1) - mu * S
    dIdt = Incidencia - (gamma + mu) * I
    dZdt = gamma * I - np.sum(TasaCont2, axis=2) - mu * Z
    dYdt = TasaCont2 - Y * (mu + gamma)

    Sv = 1 - np.sum(V, axis=1)
    H = I + np.sum(Y, axis=1)
    dVdt = np.einsum('l,lim,mi->li', Sv, delta, H) - v * V

    return Mapeo(dSdt, dIdt, dZdt, dYdt, dVdt)


__all__ = [
    # Propios de este módulo: núcleo dinámico.
    "Dimensiones", "Mapeo", "MapeoInv", "Modelo",
    "MU_DEFAULT", "GAMMA_DEFAULT", "NU_DEFAULT",
    # Reexportados desde dengue.network (construcción de la red espacial).
    "geometria_grilla", "get_idx", "build_network", "sigma_default",
    "BETA_H_DEFAULT", "BETA_V_DEFAULT", "COUPLING_DEFAULT", "SIGMA_DEFAULT",
    # Reexportado desde dengue.scenarios (condiciones iniciales).
    "condiciones_iniciales",
]
