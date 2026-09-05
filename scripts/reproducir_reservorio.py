"""Reproduce la tabla del efecto de reservorio citada en la sección 1.3 del TEX.

Compara el escenario 4 (pared de recuperados del serotipo 1, ambos serotipos
sembrados a la izquierda) con el escenario 5 (mismo foco, sin pared), para
``sigma = 1.5`` y ``sigma = 0.5``. Uso::

    .venv\\Scripts\\python.exe scripts/reproducir_reservorio.py

**Parámetros pinneados.** Los números del TEX salieron de una corrida de
scratchpad de la Sesión C1 con estas condiciones exactas; cambiarlas cambia los
decimales (por ejemplo, muestrear 300 puntos en vez de 801 mueve el tercer
decimal). Por eso van hardcodeados acá y no se toman de la CLI:

- grilla 5x9 (45 nodos; ``run.py --nodos 45`` da la misma grilla), ``c = 2``,
- ``rtol = 1e-8``, ``atol = 1e-10`` (no las tolerancias por defecto de ``solve_ivp``),
- ``t`` en ``[0, 200]`` muestreado en 801 puntos,
- el resto de los parámetros por defecto (``mu``, ``gamma``, ``nu``, ``beta``,
  ``coupling``), salvo ``sigma``.

Métricas: "pico del serotipo 2 en la mitad derecha" es el máximo, sobre el tiempo
y sobre los nodos de las columnas a la derecha de la pared, de
``I[l, 2] + sum_k Y[l, k, 2]``; "pico en la pared" es el máximo, sobre el tiempo
y sobre los nodos de la columna de la pared, del total de infectados
``sum_i I + sum_ij Y``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutarlo como `python scripts/reproducir_reservorio.py` desde la raíz del
# repo (sys.path[0] sería scripts/, no la raíz); `python -m scripts.reproducir_reservorio`
# también funciona.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.integrate import solve_ivp

from dengue.model import GAMMA_DEFAULT, MU_DEFAULT, NU_DEFAULT, Dimensiones, Mapeo, MapeoInv, Modelo
from dengue.network import build_network, get_idx, sigma_default
from dengue.scenarios import condiciones_iniciales

# --- Pinneo (ver docstring) ---------------------------------------------------
GRID_ROWS, GRID_COLS = 5, 9
C_CEPAS = 2
RTOL, ATOL = 1e-8, 1e-10
T_FINAL, N_PUNTOS = 200.0, 801
COLUMNA_PARED = GRID_COLS // 2  # 0-based, la misma que usan los escenarios 2 y 4

# Tabla del TEX (sección 1.3), a dos decimales. El script avisa si no coincide.
TABLA_TEX = {
    (1.5, 4): (0.46, 0.53),
    (1.5, 5): (0.33, None),
    (0.5, 4): (0.40, 0.34),
    (0.5, 5): (0.27, None),
}


def correr(escenario: int, sigma: float) -> dict:
    """Integra el escenario (4 o 5) con los parámetros pinneados y devuelve las métricas."""
    if escenario not in (4, 5):
        raise ValueError("este script compara los escenarios 4 (pared) y 5 (control)")
    n = GRID_ROWS * GRID_COLS
    dims = Dimensiones(n, C_CEPAS)
    N_pop = np.ones(n)
    lam, delta = build_network(GRID_ROWS, GRID_COLS, C_CEPAS)
    sig = sigma_default(C_CEPAS, sigma)
    x0 = Mapeo(*condiciones_iniciales(escenario, GRID_ROWS, GRID_COLS, C_CEPAS, N_pop))
    t_eval = np.linspace(0.0, T_FINAL, N_PUNTOS)
    sol = solve_ivp(Modelo, (0.0, T_FINAL), x0, t_eval=t_eval,
                    args=(lam, MU_DEFAULT, GAMMA_DEFAULT, sig, delta, NU_DEFAULT, N_pop, dims),
                    method='RK45', rtol=RTOL, atol=ATOL)
    assert sol.success, sol.message

    derecha = [get_idx(r, col, GRID_COLS)
               for r in range(GRID_ROWS) for col in range(COLUMNA_PARED + 1, GRID_COLS)]
    pared = [get_idx(r, COLUMNA_PARED, GRID_COLS) for r in range(GRID_ROWS)]

    pico_serotipo2_derecha = 0.0
    pico_pared = 0.0
    for k in range(sol.y.shape[1]):
        _, I, _, Y, _ = MapeoInv(sol.y[:, k], dims)
        serotipo2 = I[:, 1] + Y[:, :, 1].sum(axis=1)      # infecciosos por el serotipo 2
        total = I.sum(axis=1) + Y.sum(axis=(1, 2))         # infectados totales
        pico_serotipo2_derecha = max(pico_serotipo2_derecha, serotipo2[derecha].max())
        pico_pared = max(pico_pared, total[pared].max())

    return dict(escenario=escenario, sigma=sigma, nfev=sol.nfev,
                pico_serotipo2_derecha=float(pico_serotipo2_derecha),
                pico_pared=float(pico_pared) if escenario == 4 else None)


def tabla() -> list[dict]:
    return [correr(esc, s) for s in (1.5, 0.5) for esc in (4, 5)]


def main() -> int:
    filas = tabla()
    # Salida en ASCII a propósito: la consola de Windows (cp1252) no imprime sigma ni
    # el guion largo, y así la tabla se pega tal cual en el README o el TEX.
    print("| Configuracion | Pico serotipo 2, mitad derecha | Pico en la pared |")
    print("|---|---|---|")
    ok = True
    for f in filas:
        nombre = f"sigma={f['sigma']:g} {'con' if f['escenario'] == 4 else 'sin'} pared"
        p2 = f"{f['pico_serotipo2_derecha']:.2f}"
        pp = "-" if f['pico_pared'] is None else f"{f['pico_pared']:.2f}"
        print(f"| {nombre} | {p2} | {pp} |")
        esp2, espp = TABLA_TEX[(f['sigma'], f['escenario'])]
        coincide = (round(f['pico_serotipo2_derecha'], 2) == esp2 and
                    (espp is None or round(f['pico_pared'], 2) == espp))
        ok &= coincide
    print()
    print("Valores a cuatro decimales (grilla 5x9, c=2, rtol=1e-8, atol=1e-10, 801 puntos):")
    for f in filas:
        pp = "-" if f['pico_pared'] is None else f"{f['pico_pared']:.4f}"
        print(f"  escenario {f['escenario']}, sigma={f['sigma']:g}: serotipo 2 derecha = "
              f"{f['pico_serotipo2_derecha']:.4f}, pared = {pp}  (nfev={f['nfev']})")
    print()
    if ok:
        print("Coincide a dos decimales con la tabla de la seccion 1.3 del TEX.")
        return 0
    print("ATENCION: al menos un valor NO coincide con la tabla del TEX. No ajustar el TEX "
          "sin reconciliar el origen de la diferencia.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
