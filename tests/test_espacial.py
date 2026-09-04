"""T6 y T9 — Propiedades del acoplamiento espacial.

T6 verifica que la infección solo viaja por donde ``lam``/``delta`` lo permiten.
T9 verifica la simetría de rotación del escenario 1.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from dengue.model import (
    BETA_H_DEFAULT, BETA_V_DEFAULT, GAMMA_DEFAULT, MU_DEFAULT, NU_DEFAULT,
    Dimensiones, Mapeo, sigma_default,
)
from tests.util import TOL_ESTRICTA, armar_caso, desarmar, integrar


def sembrar_un_parche(caso: dict, parche: int, cepa: int, cantidad: float = 0.01):
    """Condición inicial libre de enfermedad con un único foco en ``parche``."""
    dims = caso["dims"]
    n, c = dims.n, dims.c
    S0 = caso["N_pop"].astype(float).copy()
    I0, Z0, Y0, V0 = np.zeros((n, c)), np.zeros((n, c)), np.zeros((n, c, c)), np.zeros((n, c))
    S0[parche] -= cantidad
    I0[parche, cepa] = cantidad
    return Mapeo(S0, I0, Z0, Y0, V0)


def assert_libre_de_enfermedad(S, I, Z, Y, V, N, parches):
    """Los ``parches`` están **exactamente** en el estado libre de enfermedad ∀ t."""
    assert_array_equal(S[parches], np.broadcast_to(N[parches][:, None], S[parches].shape))
    assert_array_equal(I[parches], 0.0)
    assert_array_equal(Z[parches], 0.0)
    assert_array_equal(Y[parches], 0.0)
    assert_array_equal(V[parches], 0.0)


def test_desacoplamiento_con_coupling_cero():
    """Con ``coupling = 0`` y un solo parche sembrado, el resto queda exactamente limpio.

    Propiedad: si ``lam[l, :, m] = delta[l, :, m] = 0`` para ``l ≠ m``, la fuerza
    de infección sobre cualquier parche no sembrado es ``0·x = 0`` en aritmética
    de punto flotante, así que ``I = Z = Y = V = 0`` y ``S = N`` se mantienen
    **exactamente** (no aproximadamente) en toda la trayectoria.
    """
    caso = armar_caso(3, 3, 2, escenario=0, coupling=0.0)
    dims, N = caso["dims"], caso["N_pop"]
    centro = 4
    x0 = sembrar_un_parche(caso, centro, cepa=0)
    sol = integrar(x0, caso["args"], **TOL_ESTRICTA)
    S, I, Z, Y, V = desarmar(sol.y, dims)

    otros = [l for l in range(dims.n) if l != centro]
    assert_libre_de_enfermedad(S, I, Z, Y, V, N, otros)
    # Cordura: el parche sembrado sí desarrolla el brote.
    assert V[centro].max() > 1e-3


def red_dirigida(c: int, beta: float, acoplamiento: float):
    """Dos parches; la infección solo puede viajar de 0 hacia 1.

    ``lam[1, :, 0]``: humanos de 1 reciben de mosquitos de 0.
    ``delta[1, :, 0]``: mosquitos de 1 se infectan de humanos de 0.
    El parche 0 no recibe nada de 1.
    """
    lam = np.zeros((2, c, 2))
    delta = np.zeros((2, c, 2))
    for l in range(2):
        lam[l, :, l] = beta
        delta[l, :, l] = beta
    lam[1, :, 0] = beta * acoplamiento
    delta[1, :, 0] = beta * acoplamiento
    return lam, delta


def test_acoplamiento_dirigido_respeta_la_direccion():
    """Con acoplamiento solo ``0 → 1``: sembrar 1 no infecta a 0; sembrar 0 sí infecta a 1.

    Propiedad: ``lam[l, i, m]`` es la fuerza **sobre** ``l`` ejercida **por**
    ``m`` (ídem ``delta``). Con ``coupling = 0`` (test anterior) las matrices
    son diagonales en ``(l, m)`` y una transposición ``l ↔ m`` pasaría
    desapercibida; acá la red es asimétrica a propósito, de modo que un einsum
    con ``'mil,mi->li'`` en vez de ``'lim,mi->li'`` (o una red construida como
    ``lam[u, :, vecino]``) haría que el parche 0 se infecte y el test falle.
    """
    c = 2
    dims = Dimensiones(2, c)
    N = np.ones(2)
    lam, delta = red_dirigida(c, BETA_H_DEFAULT, acoplamiento=0.5)
    args = (lam, MU_DEFAULT, GAMMA_DEFAULT, sigma_default(c), delta, NU_DEFAULT, N, dims)
    caso = dict(dims=dims, N_pop=N)

    # Siembra en el receptor (1): el emisor (0) queda exactamente limpio.
    sol = integrar(sembrar_un_parche(caso, 1, cepa=0), args, **TOL_ESTRICTA)
    S, I, Z, Y, V = desarmar(sol.y, dims)
    assert_libre_de_enfermedad(S, I, Z, Y, V, N, [0])
    assert V[1].max() > 1e-3

    # Siembra en el emisor (0): el receptor (1) sí se infecta.
    sol = integrar(sembrar_un_parche(caso, 0, cepa=0), args, **TOL_ESTRICTA)
    S, I, Z, Y, V = desarmar(sol.y, dims)
    assert V[1].max() > 1e-3
    assert I[1].max() > 1e-3


def rotacion_180(grid_rows: int, grid_cols: int) -> np.ndarray:
    """``rot[l]`` es el índice del nodo que queda en el lugar de ``l`` al rotar 180°."""
    return np.arange(grid_rows * grid_cols).reshape((grid_rows, grid_cols))[::-1, ::-1].ravel()


def test_simetria_del_escenario_1():
    """Escenario 1 en grilla 5×5, c=2: el campo de la cepa 0 es el de la cepa 1 rotado 180°.

    Propiedad: la grilla, ``lam``, ``delta`` y ``sigma`` son invariantes bajo la
    rotación de 180° compuesta con el intercambio de cepas ``0 ↔ 1``, y la
    condición inicial (cepa 0 arriba a la izquierda, cepa 1 abajo a la derecha)
    también. Por unicidad de la solución, la trayectoria hereda la simetría:
    ``I[l, 0](t) = I[rot(l), 1](t)``, ídem ``Z``, ``V``, ``S[l] = S[rot(l)]`` y
    ``Y[l, 0, 1] = Y[rot(l), 1, 0]``. Una asimetría en la red o en los índices
    de cepa de ``Y`` rompe la igualdad.
    """
    R, C = 5, 5
    caso = armar_caso(R, C, 2, escenario=1)
    sol = integrar(caso["x0"], caso["args"], **TOL_ESTRICTA)
    S, I, Z, Y, V = desarmar(sol.y, caso["dims"])
    rot = rotacion_180(R, C)

    assert_allclose(S, S[rot], rtol=0, atol=1e-9)
    assert_allclose(I[:, 0], I[rot, 1], rtol=0, atol=1e-9)
    assert_allclose(Z[:, 0], Z[rot, 1], rtol=0, atol=1e-9)
    assert_allclose(V[:, 0], V[rot, 1], rtol=0, atol=1e-9)
    assert_allclose(Y[:, 0, 1], Y[rot, 1, 0], rtol=0, atol=1e-9)
    # Cordura: cada cepa llegó a la esquina opuesta y hubo infección secundaria.
    assert I[rot[0], 0].max() > 1e-4
    assert Y.max() > 1e-4
