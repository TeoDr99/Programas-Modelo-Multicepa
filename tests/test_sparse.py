"""Test nuevo de la Sesión B: la ruta dispersa coincide con la densa.

No es parte del contrato congelado de la Sesión A (T1–T10) — `TAREA.md` de la
Sesión B permite explícitamente agregar este test para validar la ruta
opt-in de `dengue.sparse`. No toca ningún test existente.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from dengue.model import Dimensiones, Mapeo, Modelo
from dengue.network import build_network
from dengue.scenarios import condiciones_iniciales
from dengue.sparse import ModeloDisperso, build_network_disperso
from tests.util import TOL_ESTRICTA, integrar


def test_red_dispersa_representa_el_mismo_acoplamiento_que_la_densa():
    """``build_network_disperso`` coincide con ``build_network`` para toda cepa.

    Propiedad: la matriz dispersa `(n, n)` es la misma matriz `lam[:, i, :]`
    (o `delta[:, i, :]`) que arma la ruta densa, para cualquier cepa `i` —
    justamente la invariancia por cepa de la que depende la ruta dispersa
    (ver docstring de `dengue/sparse.py`). Además, cada fila tiene a lo sumo
    5 vecinos no nulos (self + 4 direcciones): si una futura topología
    rompiera ese patrón en silencio, este test lo vería.
    """
    R, C = 5, 5
    c = 2
    lam_densa, delta_densa = build_network(R, C, c)
    lam_disp, delta_disp = build_network_disperso(R, C, c)

    for i in range(c):
        assert_array_equal(lam_disp.toarray(), lam_densa[:, i, :])
        assert_array_equal(delta_disp.toarray(), delta_densa[:, i, :])

    nnz_por_fila = np.asarray(lam_disp.astype(bool).sum(axis=1)).ravel()
    assert nnz_por_fila.max() <= 5


def test_modelo_disperso_coincide_con_el_denso():
    """``ModeloDisperso`` integra la misma trayectoria que ``Modelo``, a 1e-10.

    Propiedad: para el mismo `x0`, la misma red (representada densa o
    dispersa) y los mismos parámetros, ambas rutas tienen que producir la
    misma dinámica. La tolerancia no es exactitud bit a bit porque sumar en
    otro orden (producto disperso vs. einsum denso) cambia los últimos bits.
    """
    R, C = 5, 5
    c = 2
    n = R * C
    dims = Dimensiones(n, c)
    N_pop = np.ones(n)
    from dengue.network import sigma_default
    sigma = sigma_default(c)

    lam_densa, delta_densa = build_network(R, C, c)
    lam_disp, delta_disp = build_network_disperso(R, C, c)

    S0, I0, Z0, Y0, V0 = condiciones_iniciales(1, R, C, c, N_pop)
    x0 = Mapeo(S0, I0, Z0, Y0, V0)

    from dengue.model import MU_DEFAULT, GAMMA_DEFAULT, NU_DEFAULT
    args_densos = (lam_densa, MU_DEFAULT, GAMMA_DEFAULT, sigma, delta_densa, NU_DEFAULT,
                   N_pop, dims)
    args_dispersos = (lam_disp, MU_DEFAULT, GAMMA_DEFAULT, sigma, delta_disp, NU_DEFAULT,
                      N_pop, dims)

    sol_densa = integrar(x0, args_densos, rhs=Modelo, **TOL_ESTRICTA)
    sol_dispersa = integrar(x0, args_dispersos, rhs=ModeloDisperso, **TOL_ESTRICTA)

    assert_allclose(sol_dispersa.y, sol_densa.y, rtol=1e-10, atol=1e-10)
    # Cordura: hubo epidemia real, no dos trayectorias triviales en cero.
    assert sol_densa.y.max() > 1e-3
