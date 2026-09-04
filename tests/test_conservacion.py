"""T2 y T3 — Conservación de la población humana.

T2 verifica la identidad algebraica sobre el lado derecho, sin integrar.
T3 verifica que la integración numérica la respeta a lo largo de una corrida.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dengue.model import Dimensiones, Mapeo, MapeoInv, Modelo
from tests.util import TOL_ESTRICTA, desarmar, integrar


def estado_admisible(rng: np.random.Generator, n: int, c: int, N: np.ndarray):
    """Estado aleatorio con ``S + ΣI + ΣZ + ΣY ≤ N`` y ``ΣV < 1`` por parche."""
    partes = rng.uniform(0.1, 1.0, size=(n, 1 + c + c + c * c))
    escala = N * rng.uniform(0.5, 1.0, n)
    partes *= (escala / partes.sum(axis=1))[:, None]
    S = partes[:, 0].copy()
    I = partes[:, 1:1 + c].copy()
    Z = partes[:, 1 + c:1 + 2 * c].copy()
    Y = partes[:, 1 + 2 * c:].reshape((n, c, c)).copy()
    V = rng.uniform(0.01, 1.0, size=(n, c))
    V *= (rng.uniform(0.3, 0.95, n) / V.sum(axis=1))[:, None]
    return S, I, Z, Y, V


@pytest.mark.parametrize("n,c", [(1, 1), (1, 3), (4, 1), (3, 2), (5, 4)])
@pytest.mark.parametrize("repeticion", range(3))
def test_identidad_de_conservacion_en_el_lado_derecho(n, c, repeticion):
    """d/dt (S_l + Σ_i I_li + Σ_i Z_li + Σ_ij Y_lij) == μ·N_l − μ·T_l − γ·Σ_ij Y_lij.

    Propiedad: la masa humana de cada parche solo entra por natalidad (``μN``),
    sale por mortalidad (``μT``) y por la recuperación de una infección
    secundaria (``γΣY``, hacia el compartimento no rastreado). Todo lo demás son
    transferencias internas entre ``S → I → Z → Y`` que deben cancelarse
    exactamente. Se evalúa el lado derecho en estados admisibles aleatorios
    (semilla fija) con ``lam``, ``delta`` y ``sigma`` densas y aleatorias, para
    que ningún einsum pueda esconderse detrás de ceros estructurales.

    Si un einsum manda masa a un compartimento equivocado (índice transpuesto,
    eje mal sumado), la identidad se rompe.
    """
    rng = np.random.default_rng(1000 * n + 10 * c + repeticion)
    dims = Dimensiones(n, c)
    N = rng.uniform(0.5, 2.0, n)
    S, I, Z, Y, V = estado_admisible(rng, n, c, N)
    lam = rng.uniform(0.0, 1.0, size=(n, c, n))
    delta = rng.uniform(0.0, 1.0, size=(n, c, n))
    sigma = rng.uniform(0.0, 2.0, size=(c, c))
    mu = rng.uniform(1e-4, 1e-2)
    gamma = rng.uniform(0.05, 0.3)
    nu = rng.uniform(0.02, 0.2)

    dx = Modelo(0.0, Mapeo(S, I, Z, Y, V), lam, mu, gamma, sigma, delta, nu, N, dims)
    dS, dI, dZ, dY, _ = MapeoInv(dx, dims)

    izquierda = dS + dI.sum(axis=1) + dZ.sum(axis=1) + dY.sum(axis=(1, 2))
    T = S + I.sum(axis=1) + Z.sum(axis=1) + Y.sum(axis=(1, 2))
    derecha = mu * N - mu * T - gamma * Y.sum(axis=(1, 2))
    assert_allclose(izquierda, derecha, rtol=1e-12, atol=1e-14)

    # Refinamiento por cepa primaria: lo que sale de Z_li por infección
    # secundaria tiene que entrar en Y_li· (mismo primer índice de cepa). La
    # identidad total de arriba no lo ve, porque suma sobre todas las cepas.
    #   dZ_li + Σ_j dY_lij == γ·I_li − μ·Z_li − (μ+γ)·Σ_j Y_lij
    izquierda_cepa = dZ + dY.sum(axis=2)
    derecha_cepa = gamma * I - mu * Z - (mu + gamma) * Y.sum(axis=2)
    assert_allclose(izquierda_cepa, derecha_cepa, rtol=1e-12, atol=1e-14)


def rhs_con_acumulador(t, xr, lam, mu, gamma, sigma, delta, nu, N, dims):
    """``Modelo`` más ``n`` variables ``R_l`` que acumulan el flujo neto de salida.

    ``dR_l/dt = γ·Σ_ij Y_lij − μ·(N_l − T_l)``, de modo que ``T_l + R_l`` es un
    invariante lineal exacto del sistema aumentado.
    """
    x = xr[:dims.total]
    dx = Modelo(t, x, lam, mu, gamma, sigma, delta, nu, N, dims)
    S, I, Z, Y, _ = MapeoInv(x, dims)
    T = S + I.sum(axis=1) + Z.sum(axis=1) + Y.sum(axis=(1, 2))
    dR = gamma * Y.sum(axis=(1, 2)) - mu * (N - T)
    return np.concatenate([dx, dR])


def test_conservacion_integrada(caso_chico):
    """T_l(t) + R_l(t) == N_l a lo largo de toda la integración (3×3, c=2, t∈[0,200]).

    Propiedad: la masa humana que sale del sistema rastreado (``γΣY``) más la
    que se pierde/gana por demografía (``μ(N−T)``) tiene que reaparecer, con
    signo cambiado, en un acumulador integrado junto con el modelo.

    Nota sobre el diseño: ``TAREA.md`` pide acumular ``γΣY`` por trapecios sobre
    la salida muestreada y comparar a 1e-6. Eso no es alcanzable: con ~300
    muestras el error de cuadratura del trapecio es del orden de 1e-2, y además
    la identidad literal ``T + ∫γΣY = N`` omite el término demográfico
    ``μ(N−T)`` (μ ≈ 3.9e-5), que sobre t=200 aporta ~1e-3. Por eso el flujo se
    acumula como estado extra del sistema de EDOs: ``T + R = N`` es entonces un
    invariante lineal que RK45 conserva a nivel de redondeo, independientemente
    de la tolerancia del integrador.
    """
    dims = caso_chico["dims"]
    N = caso_chico["N_pop"]
    xr0 = np.concatenate([caso_chico["x0"], np.zeros(dims.n)])
    sol = integrar(xr0, caso_chico["args"], rhs=rhs_con_acumulador, **TOL_ESTRICTA)

    S, I, Z, Y, _ = desarmar(sol.y[:dims.total], dims)
    R = sol.y[dims.total:]
    T = S + I.sum(axis=1) + Z.sum(axis=1) + Y.sum(axis=(1, 2))
    assert_allclose(T + R, np.broadcast_to(N[:, None], T.shape), rtol=0, atol=1e-6)

    # Cordura: el flujo acumulado es significativo, o sea que el test no es
    # trivialmente cierto porque Y se quedó en cero.
    assert Y.max() > 1e-3
    assert R[:, -1].max() > 1e-3
