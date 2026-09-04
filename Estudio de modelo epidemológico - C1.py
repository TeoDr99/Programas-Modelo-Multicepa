import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================================
# 1. CONFIGURACIÓN INTERACTIVA
# ============================================================================
print("=== CONFIGURACIÓN DEL MODELO ===")

# --- A. SELECCIÓN DE NODOS ---
try:
    raw_n = input(">> Ingrese número de nodos (ej. 100, 400, 2500): ")
    n_deseado = int(raw_n)
except ValueError:
    n_deseado = 100
    print(f"⚠️ Valor inválido. Usando defecto: {n_deseado}")

# Cálculo de geometría rectangular
root = int(np.sqrt(n_deseado))
while n_deseado % root != 0:
    root -= 1
grid_rows = root
grid_cols = n_deseado // root
n = grid_rows * grid_cols
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

# ============================================================================
# 2. GESTIÓN DE MEMORIA
# ============================================================================
size_S = n
size_I = n * c
size_Z = n * c
size_Y = n * c * c
size_V = n * c
cuts = np.cumsum([size_S, size_I, size_Z, size_Y, size_V])

def Mapeo(S, I, Z, Y, V):
    return np.concatenate([
        S.flatten(order='F'), I.flatten(order='F'), 
        Z.flatten(order='F'), Y.flatten(order='F'), V.flatten(order='F')
    ])

def MapeoInv(x):
    s, i, z, y, v = np.split(x, cuts[:-1])
    return (s, 
            i.reshape((n, c), order='F'), 
            z.reshape((n, c), order='F'), 
            y.reshape((n, c, c), order='F'), 
            v.reshape((n, c), order='F'))

# ============================================================================
# 3. MODELO DINÁMICO
# ============================================================================
def Modelo(t, x, lam, mu, gamma, sigma, delta, v, N):
    S, I, Z, Y, V = MapeoInv(x)
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
# 4. PARAMETRIZACIÓN
# ============================================================================
mu_val, gamma_val, v_val = 1.0/(70*365), 1.0/7.0, 1.0/14.0
beta_h, beta_v, coupling = 0.5, 0.5, 0.05
N_pop = np.ones(n)

# Matriz Adyacencia (Primeros Vecinos)
lam = np.zeros((n, c, n))
delta = np.zeros((n, c, n))

print("Generando red espacial...")
def get_idx(r, col): return r * grid_cols + col

for r in range(grid_rows):
    for col in range(grid_cols):
        u = get_idx(r, col)
        lam[u, :, u] = beta_h
        delta[u, :, u] = beta_v
        
        neighbors = []
        if r > 0: neighbors.append((r-1, col))
        if r < grid_rows-1: neighbors.append((r+1, col))
        if col > 0: neighbors.append((r, col-1))
        if col < grid_cols-1: neighbors.append((r, col+1))
        
        for nr, nc in neighbors:
            v_vec = get_idx(nr, nc)
            lam[v_vec, :, u] = beta_h * coupling
            delta[v_vec, :, u] = beta_v * coupling

sigma = np.ones((c, c)) * 1.5
np.fill_diagonal(sigma, 0.0)

# ============================================================================
# 5. CONDICIONES INICIALES
# ============================================================================
S0, I0 = np.array(N_pop, dtype=float), np.zeros((n, c))
Z0, Y0, V0 = np.zeros((n, c)), np.zeros((n, c, c)), np.zeros((n, c))

# 1. Choque de Ondas: Inicia dos infecciones en esquinas opuestas (Top-Left vs Bottom-Right).
#    - Objetivo: Verificar la simetría de la propagación y la competencia espacial entre dos cepas.
# 2. Cortafuegos: Crea una pared de inmunidad (Recuperados Z) en el centro del mapa.
#    - Objetivo: Verificar la física del modelo. La infección NO debe atravesar la pared.
#    - Nota: Ideal para chequear que no haya "fugas" numéricas o teletransportación.
# 3. Ruido Estocástico: Siembra múltiples focos infecciosos aleatorios por toda la red.
#    - Objetivo: Simular un brote realista desordenado y probar la estabilidad numérica con muchas cepas.

if ESCENARIO == 1: # Choque
    idx_tl = get_idx(0, 0)
    S0[idx_tl] -= 0.01; I0[idx_tl, 0] = 0.01
    if c > 1:
        idx_br = get_idx(grid_rows-1, grid_cols-1)
        S0[idx_br] -= 0.01; I0[idx_br, 1] = 0.01

elif ESCENARIO == 2: # Cortafuegos
    mid = grid_cols // 2
    for r in range(grid_rows): # Pared izquierda infectada
        idx = get_idx(r, 0)
        S0[idx] -= 0.1; I0[idx, 0] = 0.1
    for r in range(grid_rows): # Pared central inmune
        idx = get_idx(r, mid)
        S0[idx] = 0.0; Z0[idx, 0] = 1.0

elif ESCENARIO == 3: # Estocástico
    np.random.seed(42)
    for _ in range(max(1, int(n * 0.1))):
        idx = np.random.randint(0, n)
        st = np.random.randint(0, c)
        if S0[idx] > 0.01:
            S0[idx] -= 0.01; I0[idx, st] += 0.01

x0 = Mapeo(S0, I0, Z0, Y0, V0)

print("Integrando (esto puede demorar)...")
# Menos puntos temporales para acelerar animación (300 frames)
t_eval = np.linspace(0, 200, 300) 
sol = solve_ivp(Modelo, (0, 200), x0, t_eval=t_eval, 
                args=(lam, mu_val, gamma_val, sigma, delta, v_val, N_pop), 
                method='RK45')
print("Integración lista.")

# ============================================================================
# 6. VISUALIZACIÓN DE SIMULACIÓN (ANIMACIÓN)
# ============================================================================
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


