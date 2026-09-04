"""Modelo compartimental multicepa de dengue con estructura espacial.

Extracción **sin cambios semánticos** del script original
``Estudio de modelo epidemológico - C1.py`` (hoy ``run.py``): mismos ``einsum``,
mismos índices, mismos valores por defecto y mismo ``order='F'``. La única
refactorización es que ``n``, ``c`` y ``cuts`` dejan de ser variables globales
y viajan en un objeto :class:`Dimensiones`.

Este módulo no importa ``matplotlib`` a propósito: solo contiene el modelo, la
construcción de la red espacial y las condiciones iniciales por escenario, para
poder importarlos desde los tests sin abrir ninguna ventana.

Convención de índices (normativa, ver ``CLAUDE.md`` sección 2):

- ``l``, ``m`` en ``{0, ..., n-1}`` recorren parches espaciales.
- ``i``, ``j`` en ``{0, ..., c-1}`` recorren serotipos.
- ``Y[l, i, j]``: infección **primaria** por ``i``, **secundaria** por ``j``.
  Para contar humanos que infectan con la cepa ``j`` se suma sobre el eje 1.
"""

from __future__ import annotations

import numpy as np

# ============================================================================
# Parámetros por defecto del script original
# ============================================================================
MU_DEFAULT = 1.0 / (70 * 365)   # natalidad/mortalidad humana (1/días)
GAMMA_DEFAULT = 1.0 / 7.0       # tasa de recuperación humana (1/días)
NU_DEFAULT = 1.0 / 14.0         # mortalidad del mosquito (1/días)
BETA_H_DEFAULT = 0.5            # transmisión mosquito -> humano (diagonal de lam)
BETA_V_DEFAULT = 0.5            # transmisión humano -> mosquito (diagonal de delta)
COUPLING_DEFAULT = 0.05         # factor de acoplamiento con primeros vecinos
SIGMA_DEFAULT = 1.5             # efecto cruzado fuera de la diagonal (enhancement)


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


# ============================================================================
# 4. PARAMETRIZACIÓN: geometría y red espacial
# ============================================================================
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


# ============================================================================
# 5. CONDICIONES INICIALES
# ============================================================================
def condiciones_iniciales(escenario: int, grid_rows: int, grid_cols: int, c: int,
                          N_pop: np.ndarray) -> tuple[np.ndarray, ...]:
    """Devuelve ``(S0, I0, Z0, Y0, V0)`` para el escenario pedido.

    1. Choque de Ondas: dos infecciones en esquinas opuestas (cepa 0 arriba a la
       izquierda; cepa 1 abajo a la derecha si ``c > 1``).
    2. Cortafuegos: pared izquierda infectada con la cepa 0 y pared central de
       recuperados ``Z`` de la cepa 0 (ver ``CLAUDE.md`` sección 4, problema 1).
    3. Ruido Estocástico: focos aleatorios con el generador global de NumPy
       sembrado en 42 (tal cual el script original; el golden depende de esto).

    Cualquier otro valor de ``escenario`` deja el sistema sin infección, igual
    que el script original.
    """
    n = grid_rows * grid_cols
    S0, I0 = np.array(N_pop, dtype=float), np.zeros((n, c))
    Z0, Y0, V0 = np.zeros((n, c)), np.zeros((n, c, c)), np.zeros((n, c))

    if escenario == 1:  # Choque
        idx_tl = get_idx(0, 0, grid_cols)
        S0[idx_tl] -= 0.01; I0[idx_tl, 0] = 0.01
        if c > 1:
            idx_br = get_idx(grid_rows-1, grid_cols-1, grid_cols)
            S0[idx_br] -= 0.01; I0[idx_br, 1] = 0.01

    elif escenario == 2:  # Cortafuegos
        mid = grid_cols // 2
        for r in range(grid_rows):  # Pared izquierda infectada
            idx = get_idx(r, 0, grid_cols)
            S0[idx] -= 0.1; I0[idx, 0] = 0.1
        for r in range(grid_rows):  # Pared central inmune
            idx = get_idx(r, mid, grid_cols)
            S0[idx] = 0.0; Z0[idx, 0] = 1.0

    elif escenario == 3:  # Estocástico
        np.random.seed(42)
        for _ in range(max(1, int(n * 0.1))):
            idx = np.random.randint(0, n)
            st = np.random.randint(0, c)
            if S0[idx] > 0.01:
                S0[idx] -= 0.01; I0[idx, st] += 0.01

    return S0, I0, Z0, Y0, V0
