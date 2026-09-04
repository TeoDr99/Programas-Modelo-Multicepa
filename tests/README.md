# Suite de verificación del modelo

Esta suite es el contrato de la Sesión A: si alguien rompe la matemática del modelo,
algo de acá tiene que fallar. **Las sesiones siguientes no deben modificar estos tests**
(salvo regenerar el golden a propósito, ver abajo).

## Cómo correrla

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

Corre en unos 5 segundos. Ningún test importa `matplotlib` ni abre ventanas
(`test_importacion.py` lo verifica en un subproceso).

## Qué garantiza cada test

| Test | Archivo | Propiedad matemática | Qué error detecta |
|---|---|---|---|
| T1 | `test_mapeo.py` | `MapeoInv ∘ Mapeo = id` para varios `(n, c)`, incluidos `n = 1` y `c = 1`, con arrays C, Fortran y vistas no contiguas. Además fija que el vector de estado sigue orden Fortran (`x[n:2n] = I[:, 0]`). | Inconsistencias en `order='F'`, cortes mal calculados, cambio silencioso del layout del que dependen la visualización y el golden. |
| T2 | `test_conservacion.py` | Identidad puntual sobre el lado derecho: `d/dt(S + ΣI + ΣZ + ΣY) = μN − μT − γΣY`, más su refinamiento por cepa primaria `dZ_li + Σ_j dY_lij = γI_li − μZ_li − (μ+γ)Σ_j Y_lij`, con estados, `lam`, `delta` y `sigma` aleatorios densos (semilla fija), tolerancia relativa `1e-12`. | Casi cualquier error de índice en el bloque humano: un einsum que manda masa al compartimento equivocado rompe la identidad total; uno que la manda a la cepa equivocada rompe la identidad por cepa. Es el test más importante. |
| T3 | `test_conservacion.py` | Conservación integrada: con un acumulador `R_l` del flujo neto de salida como estado extra, `T_l + R_l = N_l` en toda la trayectoria (3×3, `c = 2`, `t ∈ [0, 200]`, `1e-6`). | Pérdida o creación de masa a lo largo de la integración. (Por qué no se usa trapecios: ver el docstring del test y `HALLAZGOS.md`.) |
| T4 | `test_invariantes.py` | Todos los compartimentos `≥ −1e-9` en todo `t`. | Signos invertidos, términos de salida sin su contraparte de entrada. |
| T5 | `test_invariantes.py` | `Σ_i V_li ≤ 1 + 1e-9` para todo parche y todo `t`. | Pérdida del factor `(1 − ΣV)` o suma sobre el eje equivocado en `dV`. |
| T6 | `test_espacial.py` | (a) Con `coupling = 0` y un solo parche sembrado, el resto queda **exactamente** en el estado libre de enfermedad. (b) Con una red dirigida `0 → 1`, sembrar 1 no infecta a 0 y sembrar 0 sí infecta a 1. | (a) Fugas entre parches por un eje mal sumado. (b) Transposición `l ↔ m` en `lam`/`delta` o en el einsum, que (a) no puede ver porque con `coupling = 0` las matrices son simétricas. |
| T7 | `test_una_cepa.py` | Con `c = 1`, `Y ≡ 0` exactamente y la trayectoria coincide a `1e-8` con un SIR humano + SI mosquito por parche escrito con loops explícitos, sin reutilizar `Modelo`. | Errores en la reducción a una cepa, en la máscara diagonal o en el acoplamiento humano–mosquito. |
| T8 | `test_una_cepa.py` | `n = c = 1`, `R₀ = sqrt(δλ / (ν(μ+γ)))`. Con `R₀ = 0.5` la infección decae monótonamente; con `R₀ = 2` crece, y la tasa inicial (regresión de `log I` en `t ∈ [5, 30]`) coincide al 5 % con el autovalor dominante del bloque `(I, V)` en el equilibrio libre de enfermedad. | Errores en los coeficientes lineales (`γ`, `μ`, `ν`, `λ`, `δ`) o en el acoplamiento humano–mosquito. |
| T9 | `test_espacial.py` | Escenario 1 en 5×5 con `c = 2`: el campo de la cepa 0 es el de la cepa 1 rotado 180°, en todo `t`, para `S`, `I`, `Z`, `V` e `Y` (con las cepas intercambiadas), `1e-9`. | Asimetrías en la construcción de la red, índices de cepa cruzados en `Y` o en `sigma`. |
| T10 | `test_regresion.py` | El estado final del caso `9 nodos / 2 cepas / escenario 3 / seed 42` coincide a `1e-10` con `tests/data/golden_n9_c2_esc3_seed42.npy`. | Cualquier cambio numérico, intencional o no. Es el que impide que la Sesión B cambie los números sin querer. |

## El golden

`tests/data/golden_n9_c2_esc3_seed42.npy` es el estado final `sol.y[:, -1]` de la
corrida del **script original** (antes de la extracción a `dengue/model.py`) con entradas
`9`, `2`, `3`, RK45 con tolerancias por defecto y `t_eval = linspace(0, 200, 300)`. La
trayectoria completa del módulo extraído coincide bit a bit con la del script original.

Regenerarlo **solo** si un cambio en la matemática es intencional y está documentado:

```bash
.venv\Scripts\python.exe -m tests.regenerar_golden
```

y commitear el `.npy` junto con el cambio que lo justifica.

## Convenciones

- Los tests de trayectoria (T3–T9) usan `rtol=1e-10, atol=1e-12`; T10 usa las tolerancias
  por defecto de `solve_ivp` porque reproduce la corrida original.
- Toda aleatoriedad tiene semilla fija (`np.random.default_rng(...)`).
- Helpers compartidos en `tests/util.py`; fixtures en `tests/conftest.py`.
