"""Caso de regresión del escenario 4 y regeneración deliberada de su golden.

Análogo a ``tests/regenerar_golden.py`` (T10), en archivo aparte para no tocar
el de la Sesión A. El golden ``tests/data/golden_esc4_5x9_c2_sigma15.npy`` es
el estado final ``sol.y[:, -1]`` del escenario 4 en la grilla 5x9 con ``c = 2``,
``sigma = 1.5``, RK45 con las tolerancias por defecto de ``solve_ivp`` y
``t_eval = linspace(0, 200, 300)``. Fue generado en la Sesión D, con el mismo
código que lo verifica (numpy 2.5.2, scipy 1.18.1).

Regenerar solo si un cambio del escenario 4 o de la matemática es intencional y
está documentado::

    .venv\\Scripts\\python.exe -m tests.regenerar_golden_esc4
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from dengue.model import GAMMA_DEFAULT, MU_DEFAULT, NU_DEFAULT, Dimensiones, Mapeo, Modelo
from dengue.network import build_network, sigma_default
from dengue.scenarios import condiciones_iniciales

GOLDEN = Path(__file__).parent / "data" / "golden_esc4_5x9_c2_sigma15.npy"
GRID_ROWS, GRID_COLS, C, ESCENARIO, SIGMA = 5, 9, 2, 4, 1.5


def correr_caso_golden():
    """Escenario 4, grilla 5x9, c=2, sigma=1.5, RK45 por defecto, 300 puntos en [0, 200]."""
    n = GRID_ROWS * GRID_COLS
    dims = Dimensiones(n, C)
    N_pop = np.ones(n)
    lam, delta = build_network(GRID_ROWS, GRID_COLS, C)
    sigma = sigma_default(C, SIGMA)
    x0 = Mapeo(*condiciones_iniciales(ESCENARIO, GRID_ROWS, GRID_COLS, C, N_pop))
    t_eval = np.linspace(0, 200, 300)
    sol = solve_ivp(Modelo, (0, 200), x0, t_eval=t_eval,
                    args=(lam, MU_DEFAULT, GAMMA_DEFAULT, sigma, delta, NU_DEFAULT,
                          N_pop, dims),
                    method='RK45')
    assert sol.success, sol.message
    return sol


def main() -> None:
    sol = correr_caso_golden()
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.save(GOLDEN, sol.y[:, -1])
    print(f"Golden del escenario 4 regenerado en {GOLDEN} ({sol.y.shape[0]} componentes).")


if __name__ == "__main__":
    main()
