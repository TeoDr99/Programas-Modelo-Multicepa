"""Fixtures compartidas de la suite."""

import pytest

from tests.util import TOL_ESTRICTA, armar_caso, integrar


@pytest.fixture(scope="session")
def caso_chico() -> dict:
    """Grilla 3×3, ``c = 2``, escenario 1 (choque de ondas), defaults del script."""
    return armar_caso(3, 3, 2, escenario=1)


@pytest.fixture(scope="session")
def trayectoria_chica(caso_chico):
    """Trayectoria de ``caso_chico`` en ``t ∈ [0, 200]`` con tolerancias estrictas.

    Compartida por T4 y T5 (y disponible para cualquier otro test que necesite
    una corrida de referencia con dos cepas activas).
    """
    return integrar(caso_chico["x0"], caso_chico["args"], **TOL_ESTRICTA)
