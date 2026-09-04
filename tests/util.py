"""Helpers compartidos por la suite: armado de casos e integración.

Todo lo que está acá usa los valores por defecto del script original salvo que
se indique lo contrario. Nada de acá importa ``matplotlib``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from dengue.model import (
    COUPLING_DEFAULT, GAMMA_DEFAULT, MU_DEFAULT, NU_DEFAULT,
    Dimensiones, Mapeo, MapeoInv, Modelo, build_network, condiciones_iniciales,
    sigma_default,
)

# Tolerancias estrictas para los tests de trayectoria (T3–T9). T10 usa las de
# RK45 por defecto, porque reproduce exactamente la corrida del script original.
TOL_ESTRICTA = dict(rtol=1e-10, atol=1e-12)


def armar_caso(grid_rows: int, grid_cols: int, c: int, escenario: int,
               coupling: float = COUPLING_DEFAULT, sigma: np.ndarray | None = None,
               mu: float = MU_DEFAULT, gamma: float = GAMMA_DEFAULT,
               nu: float = NU_DEFAULT) -> dict:
    """Arma un caso completo (dimensiones, red, condición inicial, ``args``)."""
    n = grid_rows * grid_cols
    dims = Dimensiones(n, c)
    N_pop = np.ones(n)
    lam, delta = build_network(grid_rows, grid_cols, c, coupling=coupling)
    if sigma is None:
        sigma = sigma_default(c)
    S0, I0, Z0, Y0, V0 = condiciones_iniciales(escenario, grid_rows, grid_cols, c, N_pop)
    x0 = Mapeo(S0, I0, Z0, Y0, V0)
    args = (lam, mu, gamma, sigma, delta, nu, N_pop, dims)
    return dict(dims=dims, x0=x0, args=args, grid=(grid_rows, grid_cols),
                N_pop=N_pop, lam=lam, delta=delta, sigma=sigma,
                mu=mu, gamma=gamma, nu=nu)


# Una integración sana con TOL_ESTRICTA usa ~3–4 mil evaluaciones del lado
# derecho. Un modelo roto (por ejemplo, un eje mal sumado que vuelve negativo un
# compartimento) puede hacer que RK45 reduzca el paso indefinidamente y la suite
# se cuelgue en vez de fallar. El presupuesto corta eso con un error claro.
MAX_NFEV = 200_000


class PresupuestoAgotado(RuntimeError):
    """El integrador superó ``MAX_NFEV`` evaluaciones: el modelo está probablemente roto."""


def integrar(x0: np.ndarray, args: tuple, t_final: float = 200.0,
             n_puntos: int = 401, rhs=Modelo, max_nfev: int = MAX_NFEV, **tol):
    """Integra con RK45 (como el script) y devuelve la solución de ``solve_ivp``."""
    contador = 0

    def rhs_acotado(t, x, *a):
        nonlocal contador
        contador += 1
        if contador > max_nfev:
            raise PresupuestoAgotado(
                f"más de {max_nfev} evaluaciones del lado derecho en t={t:.3f}; "
                "el sistema se volvió rígido o inestable (¿matemática rota?)")
        return rhs(t, x, *a)

    t_eval = np.linspace(0.0, t_final, n_puntos)
    sol = solve_ivp(rhs_acotado, (0.0, t_final), x0, t_eval=t_eval, args=args,
                    method='RK45', **tol)
    assert sol.success, sol.message
    return sol


def desarmar(y: np.ndarray, dims: Dimensiones) -> tuple[np.ndarray, ...]:
    """Convierte ``sol.y`` (forma ``(total, T)``) en ``(S, I, Z, Y, V)`` con eje temporal al final.

    Usa :func:`MapeoInv` columna a columna para no duplicar la lógica de
    ``order='F'`` en los tests.
    """
    partes = [MapeoInv(y[:, k], dims) for k in range(y.shape[1])]
    return tuple(np.stack([p[b] for p in partes], axis=-1) for b in range(5))
