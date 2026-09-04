"""Infraestructura — importar el modelo no arrastra matplotlib."""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def test_importar_el_modelo_no_carga_matplotlib():
    """``import dengue.model`` no mete ``matplotlib`` en ``sys.modules``.

    Propiedad: el módulo del modelo es puro numpy/scipy. Garantiza que ningún
    test puede abrir una ventana por accidente y que el modelo se puede usar en
    entornos sin backend gráfico. Se corre en un subproceso para que no
    interfiera lo que otros tests hayan importado.
    """
    codigo = "import sys, dengue.model; print('matplotlib' in sys.modules)"
    resultado = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ,
                               capture_output=True, text=True, check=True)
    assert resultado.stdout.strip() == "False"
