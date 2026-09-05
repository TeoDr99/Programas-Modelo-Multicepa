"""Tests nuevos de la Sesión D: escenarios 4 y 5 y el efecto de reservorio.

No son parte del contrato congelado de la Sesión A (T1–T10); `TAREA.md` de la
Sesión D los permite explícitamente. No tocan ningún test existente.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dengue.network import get_idx
from dengue.scenarios import condiciones_iniciales
from scripts.reproducir_reservorio import (
    COLUMNA_PARED, GRID_COLS, GRID_ROWS, correr,
)
from tests.regenerar_golden_esc4 import GOLDEN, correr_caso_golden

R, C = GRID_ROWS, GRID_COLS
N = np.ones(R * C)
COL_FOCO = [get_idx(r, 0, C) for r in range(R)]
COL_PARED = [get_idx(r, COLUMNA_PARED, C) for r in range(R)]
RESTO = [l for l in range(R * C) if l not in COL_FOCO and l not in COL_PARED]


def test_estructura_escenario_4():
    """El escenario 4 siembra ambos serotipos a la izquierda y pone la pared en el centro.

    Propiedad: columna 0 con I_{l1} = I_{l2} = 0.1 y S_l = 0.8; columna central
    con Z_{l1} = 1 y S_l = 0; el resto virgen (S_l = 1); Y y V nulos; masa total
    igual a 1 en cada parche. Es la condición inicial escrita en el TEX.
    """
    S, I, Z, Y, V = condiciones_iniciales(4, R, C, 2, N)
    assert_allclose(I[COL_FOCO, 0], 0.1)
    assert_allclose(I[COL_FOCO, 1], 0.1)
    assert_allclose(S[COL_FOCO], 0.8)
    assert_array_equal(Z[COL_PARED, 0], 1.0)
    assert_array_equal(Z[COL_PARED, 1], 0.0)
    assert_array_equal(S[COL_PARED], 0.0)
    assert_array_equal(S[RESTO], 1.0)
    assert_array_equal(I[RESTO], 0.0)
    assert_array_equal(Z[COL_FOCO], 0.0)
    assert_array_equal(Z[RESTO], 0.0)
    assert_array_equal(Y, 0.0)
    assert_array_equal(V, 0.0)
    assert_allclose(S + I.sum(axis=1) + Z.sum(axis=1) + Y.sum(axis=(1, 2)), 1.0)


def test_estructura_escenario_5():
    """El escenario 5 es el 4 sin pared: mismo foco, Z(0) = 0 en todas partes.

    Propiedad: idéntico al escenario 4 salvo que la columna central queda virgen.
    Es el control que hace verificable la comparación "con pared / sin pared".
    """
    S4, I4, Z4, Y4, V4 = condiciones_iniciales(4, R, C, 2, N)
    S5, I5, Z5, Y5, V5 = condiciones_iniciales(5, R, C, 2, N)
    assert_array_equal(I5, I4)
    assert_array_equal(Y5, Y4)
    assert_array_equal(V5, V4)
    assert_array_equal(Z5, 0.0)
    assert_array_equal(S5[COL_PARED], 1.0)
    otros = [l for l in range(R * C) if l not in COL_PARED]
    assert_array_equal(S5[otros], S4[otros])


@pytest.mark.parametrize("escenario", [4, 5])
def test_escenarios_4_y_5_exigen_dos_serotipos(escenario):
    """Con c = 1 los escenarios 4 y 5 fallan con un mensaje que explica por qué.

    Propiedad: el efecto de reservorio necesita un segundo serotipo que circule;
    con una sola cepa la configuración no tiene sentido y no debe degradarse en
    silencio al escenario 2.
    """
    with pytest.raises(ValueError, match="requiere c >= 2"):
        condiciones_iniciales(escenario, R, C, 1, N)


def test_escenarios_0_a_3_intactos():
    """Agregar los escenarios 4 y 5 no cambió los escenarios 0 a 3.

    Propiedad: la estructura de los escenarios 1 y 2 sigue siendo la del script
    original (el 3 está cubierto por el golden T10 y el 0 por T6). En particular
    el escenario 2 sigue sembrando SOLO el serotipo 1.
    """
    S1, I1, Z1, Y1, V1 = condiciones_iniciales(1, R, C, 2, N)
    assert I1[get_idx(0, 0, C), 0] == 0.01 and I1[get_idx(R - 1, C - 1, C), 1] == 0.01
    assert np.count_nonzero(I1) == 2 and not Z1.any()

    S2, I2, Z2, Y2, V2 = condiciones_iniciales(2, R, C, 2, N)
    assert_allclose(I2[COL_FOCO, 0], 0.1)
    assert_array_equal(I2[:, 1], 0.0)          # solo el serotipo 1
    assert_allclose(S2[COL_FOCO], 0.9)
    assert_array_equal(Z2[COL_PARED, 0], 1.0)
    assert_array_equal(S2[COL_PARED], 0.0)


@pytest.fixture(scope="module")
def tabla_reservorio():
    """Las cuatro corridas pinneadas del script de reproducción (una sola vez)."""
    return {(s, esc): correr(esc, s) for s in (1.5, 0.5) for esc in (4, 5)}


@pytest.mark.parametrize("sigma", [1.5, 0.5])
def test_efecto_reservorio(tabla_reservorio, sigma):
    """Con pared, el serotipo 2 alcanza a la derecha un pico mayor que sin pared.

    Propiedad (afirmación central de la sección 1.3 del TEX): la pared de
    recuperados del serotipo 1 no frena al serotipo 2 sino que lo amplifica,
    tanto en régimen ADE (sigma = 1.5) como con protección cruzada parcial
    (sigma = 0.5). Se compara el orden con un margen del 10 %, no el valor
    exacto, para que el test no sea frágil frente a cambios menores del
    integrador. Los valores exactos citados en el TEX los reproduce
    ``scripts/reproducir_reservorio.py``.
    """
    con = tabla_reservorio[(sigma, 4)]["pico_serotipo2_derecha"]
    sin = tabla_reservorio[(sigma, 5)]["pico_serotipo2_derecha"]
    assert 0.0 < sin < 1.0 and 0.0 < con < 1.0
    assert con > 1.1 * sin, f"sigma={sigma}: con pared {con:.4f}, sin pared {sin:.4f}"
    # La pared se infecta de verdad (son infecciones secundarias del serotipo 2).
    assert tabla_reservorio[(sigma, 4)]["pico_pared"] > 0.1


def test_efecto_reservorio_mas_fuerte_bajo_ade(tabla_reservorio):
    """La amplificación es mayor con enhancement que con protección cruzada.

    Propiedad: sigma modula la amplificación; con sigma = 1.5 el pico del
    serotipo 2 a la derecha y el pico en la pared superan a los de sigma = 0.5.
    """
    assert (tabla_reservorio[(1.5, 4)]["pico_serotipo2_derecha"]
            > tabla_reservorio[(0.5, 4)]["pico_serotipo2_derecha"])
    assert tabla_reservorio[(1.5, 4)]["pico_pared"] > tabla_reservorio[(0.5, 4)]["pico_pared"]


def test_golden_escenario_4():
    """El estado final del escenario 4 (5x9, c=2, sigma=1.5) no cambia.

    Propiedad: regresión, análoga a T10, para que sesiones futuras no alteren el
    escenario 4 sin querer. Golden en archivo aparte del de T10; regenerar solo
    a propósito con ``tests/regenerar_golden_esc4.py``.
    """
    sol = correr_caso_golden()
    esperado = np.load(GOLDEN)
    assert_allclose(sol.y[:, -1], esperado, rtol=1e-10, atol=1e-10)
