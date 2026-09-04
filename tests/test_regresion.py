"""T10 — Test de regresión contra un golden del script original."""

import numpy as np
from numpy.testing import assert_allclose

from tests.regenerar_golden import GOLDEN, correr_caso_golden


def test_regresion_golden():
    """El estado final del caso 9 nodos / 2 cepas / escenario 3 / seed 42 no cambia.

    Propiedad: determinismo y estabilidad numérica de toda la cadena
    (geometría, red, ``sigma``, condición inicial con el RNG global sembrado en
    42, ``Mapeo``, ``Modelo`` y RK45 con tolerancias por defecto). El golden fue
    generado con el script original **antes** de la extracción, así que este test
    es la prueba de que la extracción no alteró ni un bit, y es el que impide
    que la Sesión B cambie los números sin querer.

    Para regenerarlo a propósito ver ``tests/regenerar_golden.py``.
    """
    sol = correr_caso_golden()
    esperado = np.load(GOLDEN)
    assert sol.y.shape[0] == esperado.shape[0]
    assert_allclose(sol.y[:, -1], esperado, rtol=1e-10, atol=1e-10)
