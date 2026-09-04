"""Caso de regresión (T10) y regeneración deliberada del golden.

El golden ``tests/data/golden_n9_c2_esc3_seed42.npy`` es el estado final
``sol.y[:, -1]`` de la corrida ``9 nodos / 2 cepas / escenario 3`` del script
original (entradas ``9``, ``2``, ``3``), con RK45 y las tolerancias por
defecto de ``solve_ivp``, ``t_eval = linspace(0, 200, 300)``. Fue generado con
el script **anterior a la extracción** a ``dengue/model.py`` (numpy 2.5.2,
scipy 1.18.1), así que T10 certifica que la extracción no cambió los números.

Regenerar solo si un cambio de la matemática es intencional y está documentado:

    .venv\\Scripts\\python.exe -m tests.regenerar_golden

y commitear el ``.npy`` junto con el cambio que lo justifica.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from dengue.model import (
    GAMMA_DEFAULT, MU_DEFAULT, NU_DEFAULT,
    Dimensiones, Mapeo, Modelo, build_network, condiciones_iniciales,
    geometria_grilla, sigma_default,
)

GOLDEN = Path(__file__).parent / "data" / "golden_n9_c2_esc3_seed42.npy"
N_DESEADO, C, ESCENARIO = 9, 2, 3


def correr_caso_golden():
    """Reproduce exactamente la corrida del script original con entradas 9 / 2 / 3."""
    grid_rows, grid_cols, n = geometria_grilla(N_DESEADO)
    dims = Dimensiones(n, C)
    N_pop = np.ones(n)
    lam, delta = build_network(grid_rows, grid_cols, C)
    sigma = sigma_default(C)
    x0 = Mapeo(*condiciones_iniciales(ESCENARIO, grid_rows, grid_cols, C, N_pop))
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
    print(f"Golden regenerado en {GOLDEN} ({sol.y.shape[0]} componentes).")


if __name__ == "__main__":
    main()
