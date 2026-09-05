"""Simulación interactiva del modelo multicepa de dengue con estructura espacial.

Este es el script original (``Estudio de modelo epidemológico - C1.py``) con el
modelo movido a ``dengue/model.py``. Pide por consola el número de nodos, de
cepas y el escenario, integra el sistema y muestra una animación del total de
infectados sobre la grilla.
"""

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


def main() -> None:
    # ========================================================================
    # 1. CONFIGURACIÓN INTERACTIVA
    # ========================================================================
    print("=== CONFIGURACIÓN DEL MODELO ===")

    # --- A. SELECCIÓN DE NODOS ---
    try:
        raw_n = input(">> Ingrese número de nodos (ej. 100, 400, 2500): ")
        n_deseado = int(raw_n)
    except ValueError:
        n_deseado = 100
        print(f"⚠️ Valor inválido. Usando defecto: {n_deseado}")

    # Cálculo de geometría rectangular
    grid_rows, grid_cols, n = geometria_grilla(n_deseado)
    print(f"-> Topología: {grid_rows}x{grid_cols} ({n} nodos)")

    # --- B. SELECCIÓN DE CEPAS ---
    try:
        raw_c = input(">> Ingrese número de cepas (ej. 1, 2, 10): ")
        c = int(raw_c)
        if c < 1: raise ValueError
    except ValueError:
        c = 2
        print(f"⚠️ Valor inválido. Usando defecto: {c}")

    print("\nEscenarios:")
    print("   1: Choque de Ondas (Simetría)")
    print("   2: Cortafuegos (Test de Barrera)")
    print("   3: Ruido Estocástico (Aleatorio)")
    try:
        raw_esc = input(">> Elija Escenario (1-3): ")
        ESCENARIO = int(raw_esc)
    except ValueError:
        ESCENARIO = 3
        print("⚠️ Usando defecto: 3")

    # ========================================================================
    # 2. GESTIÓN DE MEMORIA
    # ========================================================================
    dims = Dimensiones(n, c)

    # ========================================================================
    # 4. PARAMETRIZACIÓN
    # ========================================================================
    mu_val, gamma_val, v_val = MU_DEFAULT, GAMMA_DEFAULT, NU_DEFAULT
    beta_h, beta_v, coupling = BETA_H_DEFAULT, BETA_V_DEFAULT, COUPLING_DEFAULT
    N_pop = np.ones(n)

    # Matriz Adyacencia (Primeros Vecinos)
    print("Generando red espacial...")
    lam, delta = build_network(grid_rows, grid_cols, c, beta_h, beta_v, coupling)
    sigma = sigma_default(c)

    # ========================================================================
    # 5. CONDICIONES INICIALES
    # ========================================================================
    # 1. Choque de Ondas: Inicia dos infecciones en esquinas opuestas (Top-Left vs Bottom-Right).
    #    - Objetivo: Verificar la simetría de la propagación y la competencia espacial entre dos cepas.
    # 2. Cortafuegos: Crea una pared de inmunidad (Recuperados Z) en el centro del mapa.
    #    - Objetivo: Verificar la física del modelo. La infección NO debe atravesar la pared.
    #    - Nota: Ideal para chequear que no haya "fugas" numéricas o teletransportación.
    # 3. Ruido Estocástico: Siembra múltiples focos infecciosos aleatorios por toda la red.
    #    - Objetivo: Simular un brote realista desordenado y probar la estabilidad numérica con muchas cepas.
    S0, I0, Z0, Y0, V0 = condiciones_iniciales(ESCENARIO, grid_rows, grid_cols, c, N_pop)
    x0 = Mapeo(S0, I0, Z0, Y0, V0)

    print("Integrando (esto puede demorar)...")
    # Menos puntos temporales para acelerar animación (300 frames)
    t_eval = np.linspace(0, 200, 300)
    sol = solve_ivp(Modelo, (0, 200), x0, t_eval=t_eval,
                    args=(lam, mu_val, gamma_val, sigma, delta, v_val, N_pop, dims),
                    method='RK45')
    print("Integración lista.")

    # ========================================================================
    # 6. VISUALIZACIÓN DE SIMULACIÓN (ANIMACIÓN)
    # ========================================================================
    print("Generando animación...")

    # Recuperar Infectados Primarios (I) + Secundarios (Y)
    # I shape: (n, c, t), Y shape: (n, c, c, t)
    idx_S, idx_I = n, n + n*c
    idx_Z, idx_Y = idx_I + n*c, idx_I + n*c + n*c*c

    I_res = sol.y[idx_S:idx_I].reshape((n, c, len(t_eval)), order='F')
    Y_res = sol.y[idx_Z:idx_Y].reshape((n, c, c, len(t_eval)), order='F')

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
