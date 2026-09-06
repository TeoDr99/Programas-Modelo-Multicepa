"""Condiciones iniciales por escenario.

Movido verbatim desde ``dengue/model.py`` (Sesión B, B1): sin cambios de
comportamiento, incluida la llamada a ``np.random.seed(42)`` del escenario 3
(congelada a propósito, ver ``HALLAZGOS.md``).
"""

from __future__ import annotations

import numpy as np

from .network import get_idx


def condiciones_iniciales(escenario: int, grid_rows: int, grid_cols: int, c: int,
                          N_pop: np.ndarray) -> tuple[np.ndarray, ...]:
    """Devuelve ``(S0, I0, Z0, Y0, V0)`` para el escenario pedido.

    0. Estado libre de enfermedad: ``S0 = N``, el resto en cero. No es un
       accidente del script original (ver hallazgo 8) sino un valor explícito
       de la CLI (``--escenario 0``) que T6 usa para partir de un estado limpio
       y sembrar a mano.
    1. Choque de Ondas: dos infecciones en esquinas opuestas (cepa 0 arriba a la
       izquierda; cepa 1 abajo a la derecha si ``c > 1``).
    2. Cortafuegos: pared izquierda infectada con la cepa 0 y pared central de
       recuperados ``Z`` de la cepa 0 (ver ``CLAUDE.md`` sección 4, problema 1).
    3. Ruido Estocástico: focos aleatorios con el generador global de NumPy
       sembrado en 42 (tal cual el script original; el golden depende de esto).
    4. Reservorio: la misma pared central de recuperados del serotipo 1 que el
       escenario 2, pero con **ambos** serotipos sembrados en la columna
       izquierda (``eps = 0.1`` cada uno). Existe para exhibir el efecto de
       reservorio: los recuperados del serotipo 1 quedan fuera de la competencia
       por susceptibles y le llegan intactos al serotipo 2, que atraviesa la
       pared amplificado (sección 1.3 del TEX, hallazgo 24). El escenario 2 es
       su versión de un solo serotipo, en la que el efecto no puede aparecer.
       Requiere ``c >= 2``.
    5. Control del reservorio: idéntico al 4 pero sin la pared (``Z(0) = 0``),
       para que la comparación "con pared / sin pared" sea reproducible.
       Requiere ``c >= 2``.

    Cualquier valor de ``escenario`` fuera de ``{1, ..., 5}`` deja el sistema sin
    infección, igual que el script original (de ahí que ``0`` sea válido).
    Llamar a esta función con los escenarios 0 a 3 devuelve exactamente lo mismo
    que antes de agregar el 4 y el 5 (Sesión D).
    """
    n = grid_rows * grid_cols
    S0, I0 = np.array(N_pop, dtype=float), np.zeros((n, c))
    Z0, Y0, V0 = np.zeros((n, c)), np.zeros((n, c, c)), np.zeros((n, c))

    if escenario == 1:  # Choque
        idx_tl = get_idx(0, 0, grid_cols)
        S0[idx_tl] -= 0.01; I0[idx_tl, 0] = 0.01
        if c > 1:
            idx_br = get_idx(grid_rows-1, grid_cols-1, grid_cols)
            S0[idx_br] -= 0.01; I0[idx_br, 1] = 0.01

    elif escenario == 2:  # Cortafuegos
        mid = grid_cols // 2
        for r in range(grid_rows):  # Pared izquierda infectada
            idx = get_idx(r, 0, grid_cols)
            S0[idx] -= 0.1; I0[idx, 0] = 0.1
        for r in range(grid_rows):  # Pared central inmune
            idx = get_idx(r, mid, grid_cols)
            S0[idx] = 0.0; Z0[idx, 0] = 1.0

    elif escenario == 3:  # Estocástico
        np.random.seed(42)
        for _ in range(max(1, int(n * 0.1))):
            idx = np.random.randint(0, n)
            st = np.random.randint(0, c)
            if S0[idx] > 0.01:
                S0[idx] -= 0.01; I0[idx, st] += 0.01

    elif escenario in (4, 5):  # Reservorio (4: con pared; 5: control sin pared)
        if c < 2:
            raise ValueError(
                f"El escenario {escenario} requiere c >= 2: exhibe el efecto de reservorio "
                "de una pared de recuperados del serotipo 1 sobre el serotipo 2; con una "
                "sola cepa no hay segundo serotipo que circule (usar el escenario 2).")
        for r in range(grid_rows):  # Columna izquierda: ambos serotipos
            idx = get_idx(r, 0, grid_cols)
            S0[idx] -= 0.1; I0[idx, 0] = 0.1
            S0[idx] -= 0.1; I0[idx, 1] = 0.1
        if escenario == 4:
            mid = grid_cols // 2
            for r in range(grid_rows):  # Pared central inmune al serotipo 1
                idx = get_idx(r, mid, grid_cols)
                S0[idx] = 0.0; Z0[idx, 0] = 1.0

    return S0, I0, Z0, Y0, V0
