"""T1 — Ida y vuelta del mapeo entre bloques ``(S, I, Z, Y, V)`` y el vector de estado."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dengue.model import Dimensiones, Mapeo, MapeoInv


def bloques_distintos(n: int, c: int) -> tuple[np.ndarray, ...]:
    """Bloques con un valor distinto en cada posición (ni ceros ni unos)."""
    S = 1000.0 + 1.25 * np.arange(n)
    I = 2000.0 + 1.25 * np.arange(n * c).reshape((n, c))
    Z = 3000.0 + 1.25 * np.arange(n * c).reshape((n, c))
    Y = 4000.0 + 1.25 * np.arange(n * c * c).reshape((n, c, c))
    V = 5000.0 + 1.25 * np.arange(n * c).reshape((n, c))
    return S, I, Z, Y, V


CASOS = [(1, 1), (1, 3), (4, 1), (3, 2), (6, 4)]


@pytest.mark.parametrize("n,c", CASOS)
def test_ida_y_vuelta(n, c):
    """MapeoInv(Mapeo(S, I, Z, Y, V)) devuelve exactamente los bloques originales.

    Propiedad: ``MapeoInv ∘ Mapeo = id`` sobre bloques de formas ``(n,)``,
    ``(n,c)``, ``(n,c)``, ``(n,c,c)``, ``(n,c)``, incluyendo ``n = 1`` y ``c = 1``.
    Con valores distintos en cada posición, cualquier inconsistencia en el orden
    de aplanado (``order='F'`` vs ``'C'``) entre ida y vuelta se detecta.
    """
    dims = Dimensiones(n, c)
    bloques = bloques_distintos(n, c)
    x = Mapeo(*bloques)
    assert x.shape == (dims.total,)
    recuperados = MapeoInv(x, dims)
    for original, recuperado in zip(bloques, recuperados):
        assert recuperado.shape == original.shape
        assert_array_equal(recuperado, original)


@pytest.mark.parametrize("n,c", CASOS)
def test_ida_y_vuelta_con_arrays_fortran_y_vistas(n, c):
    """La ida y vuelta no depende de cómo estén almacenados los arrays en memoria.

    Propiedad: ``Mapeo`` produce el mismo vector para el mismo contenido lógico,
    sea el array C-contiguo, F-contiguo o una vista no contigua.
    """
    dims = Dimensiones(n, c)
    bloques = bloques_distintos(n, c)
    x_c = Mapeo(*bloques)
    x_f = Mapeo(*[np.asfortranarray(b) for b in bloques])
    # Vistas no contiguas: cada bloque es un slice con paso 2 de un array mayor.
    vistas = []
    for b in bloques:
        grande = np.zeros(tuple(2 * s for s in b.shape))
        grande[tuple(slice(None, None, 2) for _ in b.shape)] = b
        vistas.append(grande[tuple(slice(None, None, 2) for _ in b.shape)])
    x_v = Mapeo(*vistas)
    assert_array_equal(x_f, x_c)
    assert_array_equal(x_v, x_c)
    for original, recuperado in zip(bloques, MapeoInv(x_f, dims)):
        assert_array_equal(recuperado, original)


def test_layout_es_orden_fortran():
    """El vector de estado sigue la convención Fortran del script original.

    Propiedad: dentro del bloque ``I`` los primeros ``n`` elementos son
    ``I[:, 0]`` (la cepa 0 de todos los parches), no ``I[0, :]``. Esta es la
    convención de la que dependen la visualización de ``run.py`` y el golden de
    T10; la ida y vuelta sola no la fija (un par ``'C'``/``'C'`` también sería
    consistente).
    """
    n, c = 3, 2
    dims = Dimensiones(n, c)
    S, I, Z, Y, V = bloques_distintos(n, c)
    x = Mapeo(S, I, Z, Y, V)
    assert_array_equal(x[:n], S)
    assert_array_equal(x[n:2 * n], I[:, 0])
    assert_array_equal(x[2 * n:3 * n], I[:, 1])
    inicio_Y = dims.cuts[2]
    # Y[l, i, j] aplanado en Fortran: l varía más rápido, luego i, luego j.
    assert_array_equal(x[inicio_Y:inicio_Y + n], Y[:, 0, 0])
    assert_array_equal(x[inicio_Y + n:inicio_Y + 2 * n], Y[:, 1, 0])
