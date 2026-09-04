"""T7 y T8 — Reducción a una sola cepa y umbral epidémico."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp

from dengue.model import GAMMA_DEFAULT, MU_DEFAULT, NU_DEFAULT, Dimensiones, Mapeo, Modelo
from tests.util import TOL_ESTRICTA, armar_caso, desarmar, integrar


# ----------------------------------------------------------------------------
# T7
# ----------------------------------------------------------------------------
def rhs_sir_vector(t, u, lam2, delta2, mu, gamma, nu, N, n):
    """Referencia independiente: SIR humano + SI mosquito por parche, con loops.

    ``u = [S_0..S_{n-1}, I_0.., Z_0.., V_0..]``. ``lam2[l, m]`` es la fuerza
    sobre humanos de ``l`` por mosquitos de ``m``; ``delta2[l, m]`` la infección
    de mosquitos de ``l`` por humanos de ``m``. No usa ``Mapeo`` ni ``Modelo``.
    """
    S, I, Z, V = u[:n], u[n:2 * n], u[2 * n:3 * n], u[3 * n:]
    dS, dI, dZ, dV = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for l in range(n):
        fuerza = 0.0
        for m in range(n):
            fuerza += lam2[l, m] * V[m]
        dS[l] = mu * N[l] - S[l] * fuerza - mu * S[l]
        dI[l] = S[l] * fuerza - (gamma + mu) * I[l]
        dZ[l] = gamma * I[l] - mu * Z[l]
        contagio = 0.0
        for m in range(n):
            contagio += delta2[l, m] * I[m]
        dV[l] = (1.0 - V[l]) * contagio - nu * V[l]
    return np.concatenate([dS, dI, dZ, dV])


def test_reduccion_a_una_cepa():
    """Con ``c = 1``, ``Y ≡ 0`` y la dinámica coincide con un SIR-vector escrito aparte.

    Propiedad: con una sola cepa no hay infección secundaria (``mask = 0`` anula
    ``TasaCont2`` aunque ``sigma`` tenga diagonal no nula), y el modelo se reduce por parche a
    ``S → I → Z`` en humanos y ``S_v → V`` en mosquitos. La referencia se
    integra con el mismo solver y las mismas tolerancias; la comparación es a
    ``1e-8`` absoluto. ``Y`` se compara con cero **exactamente**.
    """
    R, C = 3, 3
    # sigma con diagonal NO nula: la única barrera contra la reinfección es mask.
    caso = armar_caso(R, C, 1, escenario=1, sigma=np.full((1, 1), 1.5))
    dims, N = caso["dims"], caso["N_pop"]
    n = dims.n
    sol = integrar(caso["x0"], caso["args"], **TOL_ESTRICTA)
    S, I, Z, Y, V = desarmar(sol.y, dims)
    assert_array_equal(Y, 0.0)

    lam2, delta2 = caso["lam"][:, 0, :], caso["delta"][:, 0, :]
    S0, I0, Z0, _, V0 = (b.reshape(n) for b in
                          (S[:, 0], I[:, 0, 0], Z[:, 0, 0], Y[:, 0, 0, 0], V[:, 0, 0]))
    u0 = np.concatenate([S0, I0, Z0, V0])
    ref = solve_ivp(rhs_sir_vector, (0.0, 200.0), u0, t_eval=sol.t,
                    args=(lam2, delta2, caso["mu"], caso["gamma"], caso["nu"], N, n),
                    method='RK45', **TOL_ESTRICTA)
    assert ref.success, ref.message

    assert_allclose(S, ref.y[:n], rtol=0, atol=1e-8)
    assert_allclose(I[:, 0], ref.y[n:2 * n], rtol=0, atol=1e-8)
    assert_allclose(Z[:, 0], ref.y[2 * n:3 * n], rtol=0, atol=1e-8)
    assert_allclose(V[:, 0], ref.y[3 * n:], rtol=0, atol=1e-8)
    # Cordura: hubo epidemia (el test no compara dos trayectorias triviales).
    assert I.max() > 0.05


# ----------------------------------------------------------------------------
# T8
# ----------------------------------------------------------------------------
def beta_para_R0(R0: float, mu: float, gamma: float, nu: float) -> float:
    """``λ = δ = β`` tal que ``R₀ = sqrt(δλ / (ν(μ+γ)))`` valga ``R0``."""
    return R0 * np.sqrt(nu * (mu + gamma))


def jacobiano_dfe(beta: float, mu: float, gamma: float, nu: float) -> np.ndarray:
    """Bloque ``(I, V)`` de la linealización alrededor del equilibrio libre de enfermedad."""
    return np.array([[-(gamma + mu), beta],
                     [beta, -nu]])


def modo_dominante(J: np.ndarray):
    """Autovalor dominante (mayor parte real) y su autovector normalizado, positivo."""
    w, U = np.linalg.eig(J)
    k = np.argmax(w.real)
    vec = U[:, k].real
    vec = np.abs(vec) / np.abs(vec).sum()
    return w[k].real, vec


def correr_un_parche_una_cepa(beta: float, mu: float, gamma: float, nu: float,
                              semilla: float = 1e-6, t_final: float = 200.0):
    """Integra ``n = c = 1`` sembrando ``semilla`` sobre el modo dominante ``(I, V)``."""
    dims = Dimensiones(1, 1)
    N = np.ones(1)
    lam = np.full((1, 1, 1), beta)
    delta = np.full((1, 1, 1), beta)
    sigma = np.zeros((1, 1))
    r, vec = modo_dominante(jacobiano_dfe(beta, mu, gamma, nu))
    I0, V0 = semilla * vec
    x0 = Mapeo(np.array([1.0 - I0]), np.array([[I0]]), np.zeros((1, 1)),
               np.zeros((1, 1, 1)), np.array([[V0]]))
    sol = integrar(x0, (lam, mu, gamma, sigma, delta, nu, N, dims), t_final=t_final,
                   n_puntos=2001, **TOL_ESTRICTA)
    S, I, Z, Y, V = desarmar(sol.y, dims)
    return sol.t, I[0, 0], V[0, 0], r


def test_umbral_subcritico_decae():
    """Con ``R₀ = 0.5`` la infección decae monótonamente a cero (``n = c = 1``).

    Propiedad: si ``R₀ < 1`` el equilibrio libre de enfermedad es localmente
    estable (ambos autovalores del bloque ``(I, V)`` son negativos). Sembrando
    sobre el modo dominante, ``I`` y ``V`` decaen de forma monótona.
    """
    mu, gamma, nu = MU_DEFAULT, GAMMA_DEFAULT, NU_DEFAULT
    beta = beta_para_R0(0.5, mu, gamma, nu)
    t, I, V, r = correr_un_parche_una_cepa(beta, mu, gamma, nu)
    assert r < 0
    assert np.all(np.diff(I) <= 1e-9 * I[0])
    assert np.all(np.diff(V) <= 1e-9 * V[0])
    assert I[-1] < 1e-3 * I[0]
    assert V[-1] < 1e-3 * V[0]


def test_umbral_supercritico_crece_con_la_tasa_predicha():
    """Con ``R₀ = 2`` la infección crece, y a la tasa del autovalor dominante (5 %).

    Propiedad: si ``R₀ > 1`` el equilibrio libre de enfermedad es inestable y,
    mientras ``S ≈ N`` y ``Σ V ≪ 1``, ``I(t) ∝ exp(r·t)`` con ``r`` el autovalor
    dominante de ``[[−(γ+μ), λ], [δ, −ν]]``. Se estima ``r`` por regresión
    lineal de ``log I`` en ``t ∈ [5, 30]`` y se compara con tolerancia relativa
    del 5 %. Un error en el acoplamiento humano–mosquito cambia ``r``.
    """
    mu, gamma, nu = MU_DEFAULT, GAMMA_DEFAULT, NU_DEFAULT
    beta = beta_para_R0(2.0, mu, gamma, nu)
    t, I, V, r = correr_un_parche_una_cepa(beta, mu, gamma, nu)
    assert r > 0
    ventana = (t >= 5.0) & (t <= 30.0)
    pendiente, _ = np.polyfit(t[ventana], np.log(I[ventana]), 1)
    assert_allclose(pendiente, r, rtol=0.05)
    # Crece desde una siembra chica hasta un brote macroscópico.
    assert I.max() > 1e3 * I[0]
    assert I.max() > 1e-2
