"""Simulación del modelo multicepa de dengue con estructura espacial.

Punto de entrada de línea de comandos. Arma la red espacial, integra el
sistema de EDOs y anima el total de infectados sobre la grilla. Ejecutar
``python run.py --help`` para ver las opciones.
"""

import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dengue.model import Dimensiones, Mapeo, Modelo, MU_DEFAULT, GAMMA_DEFAULT, NU_DEFAULT
from dengue.network import (
    build_network, geometria_grilla, sigma_default,
    BETA_H_DEFAULT, BETA_V_DEFAULT, COUPLING_DEFAULT,
)
from dengue.scenarios import condiciones_iniciales
from dengue.sparse import ModeloDisperso, build_network_disperso

SEMILLA_ESCENARIO_3 = 42  # ver HALLAZGOS.md: congelada a propósito, el golden depende de ella.

EPILOGO_ESCENARIOS = """\
Escenarios (--escenario):
  0: Estado limpio, sin siembra. Útil como punto de partida para sembrar a mano.
  1: Choque de Ondas. Dos infecciones en esquinas opuestas: simetría y competencia espacial.
  2: Cortafuegos. Pared de inmunidad en el centro (ver limitaciones en el README: con
     sigma > 1 la pared no aísla, es un resultado del modelo, no un bug).
  3: Ruido Estocástico (default). Focos aleatorios; usa una semilla fija internamente.
"""


def entero_positivo(valor: str) -> int:
    """Valida que ``valor`` sea un entero >= 1 (para --nodos y --cepas)."""
    try:
        ivalor = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"debe ser un entero, se recibió: {valor!r}")
    if ivalor < 1:
        raise argparse.ArgumentTypeError(f"debe ser >= 1, se recibió: {ivalor}")
    return ivalor


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simula el modelo multicepa de dengue con estructura espacial.",
        epilog=EPILOGO_ESCENARIOS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodos", type=entero_positivo, default=100,
                        help="número de nodos de la grilla (default: 100)")
    parser.add_argument("--cepas", type=entero_positivo, default=2,
                        help="número de serotipos (default: 2)")
    parser.add_argument("--escenario", type=int, choices=[0, 1, 2, 3], default=3,
                        help="condición inicial a usar (default: 3; ver detalle abajo)")
    parser.add_argument("--t-final", type=float, default=200.0, dest="t_final",
                        help="tiempo final de la integración, en días (default: 200)")
    parser.add_argument("--semilla", type=int, default=SEMILLA_ESCENARIO_3,
                        help="semilla del RNG (default: 42). Hoy no cambia ningún resultado: "
                             "los escenarios 0/1/2 son determinísticos y el escenario 3 tiene "
                             "su semilla congelada en 42 (ver HALLAZGOS.md). Se acepta por "
                             "completitud de la CLI y para escenarios aleatorios futuros")
    parser.add_argument("--sparse", action="store_true",
                        help="usar el acoplamiento disperso (más rápido para --nodos grande; "
                             "ver la sección de rendimiento del README). Apagado por defecto")
    return parser


def main() -> None:
    args = construir_parser().parse_args()

    if args.escenario == 3 and args.semilla != SEMILLA_ESCENARIO_3:
        print(f"aviso: --semilla {args.semilla} se ignora en el escenario 3, cuya semilla "
              f"está congelada en {SEMILLA_ESCENARIO_3} (ver HALLAZGOS.md)", file=sys.stderr)

    grid_rows, grid_cols, n = geometria_grilla(args.nodos)
    c = args.cepas
    print(f"-> Topología: {grid_rows}x{grid_cols} ({n} nodos), {c} cepa(s), "
          f"escenario {args.escenario}")

    dims = Dimensiones(n, c)

    mu_val, gamma_val, v_val = MU_DEFAULT, GAMMA_DEFAULT, NU_DEFAULT
    beta_h, beta_v, coupling = BETA_H_DEFAULT, BETA_V_DEFAULT, COUPLING_DEFAULT
    N_pop = np.ones(n)

    print(f"Generando red espacial ({'dispersa' if args.sparse else 'densa'})...")
    if args.sparse:
        lam, delta = build_network_disperso(grid_rows, grid_cols, c, beta_h, beta_v, coupling)
        rhs = ModeloDisperso
    else:
        lam, delta = build_network(grid_rows, grid_cols, c, beta_h, beta_v, coupling)
        rhs = Modelo
    sigma = sigma_default(c)

    S0, I0, Z0, Y0, V0 = condiciones_iniciales(args.escenario, grid_rows, grid_cols, c, N_pop)
    x0 = Mapeo(S0, I0, Z0, Y0, V0)

    print("Integrando (esto puede demorar)...")
    # Menos puntos temporales para acelerar animación (300 frames)
    t_eval = np.linspace(0, args.t_final, 300)
    sol = solve_ivp(rhs, (0, args.t_final), x0, t_eval=t_eval,
                    args=(lam, mu_val, gamma_val, sigma, delta, v_val, N_pop, dims),
                    method='RK45')
    print("Integración lista.")

    # ========================================================================
    # VISUALIZACIÓN DE SIMULACIÓN (ANIMACIÓN)
    # ========================================================================
    print("Generando animación...")

    # Recuperar Infectados Primarios (I) + Secundarios (Y) usando los mismos
    # cortes que MapeoInv (hallazgo 10: los nombres idx_S/idx_Z de la versión
    # anterior eran engañosos — daban bien por cancelación de offsets, pero
    # apuntaban al inicio de I y de Y, no de S ni de Z).
    # I shape: (n, c, t), Y shape: (n, c, c, t)
    cuts = dims.cuts
    I_res = sol.y[cuts[0]:cuts[1]].reshape((n, c, len(t_eval)), order='F')
    Y_res = sol.y[cuts[2]:cuts[3]].reshape((n, c, c, len(t_eval)), order='F')

    # Sumar todas las cepas y tipos de infección para obtener "Carga Viral Total"
    # Total = Sum(I) + Sum(Y) a lo largo de cepas
    I_total = np.sum(I_res, axis=1) # (n, t)
    Y_total = np.sum(np.sum(Y_res, axis=1), axis=1) # (n, t)
    Infectados_Total = I_total + Y_total # (n, t)

    # Preparar figura
    fig, ax = plt.subplots(figsize=(8, 6))
    max_val = np.max(Infectados_Total)

    # Imagen inicial
    grid_data = Infectados_Total[:, 0].reshape((grid_rows, grid_cols))
    im = ax.imshow(grid_data, cmap='inferno', vmin=0, vmax=max_val, interpolation='nearest')
    plt.colorbar(im, label='Densidad Total de Infectados')
    title = ax.set_title(f'Simulación t=0.0')

    def update(frame):
        # Actualizar datos de la grilla para el tiempo t[frame]
        grid_data = Infectados_Total[:, frame].reshape((grid_rows, grid_cols))
        im.set_data(grid_data)
        title.set_text(f'Expansión - Día {t_eval[frame]:.1f}')
        return im, title

    anim = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=False)

    plt.show()


if __name__ == "__main__":
    main()
