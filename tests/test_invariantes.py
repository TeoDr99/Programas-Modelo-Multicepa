"""T4 y T5 — Invariantes de signo y de cota sobre una trayectoria."""

from tests.util import desarmar


def test_positividad(trayectoria_chica):
    """Todos los compartimentos se mantienen ≥ −1e-9 en todo t.

    Propiedad: el ortante no negativo es invariante para el sistema (cada
    derivada es no negativa en la frontera donde su variable vale cero). Un
    einsum con signo o índice equivocado suele violar esto rápido.
    """
    assert trayectoria_chica.y.min() >= -1e-9


def test_cota_de_mosquitos(trayectoria_chica, caso_chico):
    """Σ_i V_li ≤ 1 + 1e-9 para todo parche l y todo t.

    Propiedad: el factor ``(1 − Σ_i V_li)`` en ``dV`` hace invariante el simplex
    de mosquitos: en ``ΣV = 1`` la derivada de la suma vale ``−ν·ΣV < 0``.
    """
    _, _, _, _, V = desarmar(trayectoria_chica.y, caso_chico["dims"])
    assert V.sum(axis=1).max() <= 1 + 1e-9
    # Cordura: la población de mosquitos se infectó de verdad.
    assert V.max() > 1e-3
