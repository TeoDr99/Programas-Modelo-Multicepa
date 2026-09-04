# CLAUDE.md — Modelo multicepa de dengue con estructura espacial

Este archivo es contexto permanente del proyecto. La tarea concreta de cada sesión
está en `TAREA.md`. Si hay conflicto entre ambos, manda `CLAUDE.md`.

---

## 1. Qué es este proyecto

Implementación numérica de un modelo compartimental de transmisión de dengue con
**cuatro serotipos (generalizable a `c` cepas)** y **`n` parches espaciales**.

- Es una generalización del modelo de **de Araújo et al. (2023)**, *"Applying a
  multi-strain dengue model to epidemics data"*, Mathematical Biosciences 360, 109013.
  Ese paper tiene 25 EDOs y un solo parche; acá hay 25 EDOs por parche.
- Surge del proyecto **"Estudio de modelos matemáticos estructurados para la
  transmisión del dengue"** (Becas de Ayudantía de Investigación UNRC-ODS,
  Departamento de Matemática, Universidad Nacional de Río Cuarto, Argentina),
  dentro del proyecto marco *"Problemas teóricos en ecuaciones diferenciales y
  cálculo de variaciones"*, dirigido por Fernando Mazzone.
- Existe un documento LaTeX acompañante con la formulación matemática y un teorema
  de existencia de equilibrio endémico. **El código y el TEX deben mantenerse
  consistentes entre sí.**

**Destino del repositorio:** público en GitHub, enlazado desde el CV del autor en la
sección de experiencia en investigación. El estándar de calidad es "un revisor
técnico abre esto y en dos minutos entiende qué hace, cómo correrlo y por qué
confiar en los resultados".

---

## 2. Modelo: variables de estado y convención de índices

**Esta sección es normativa. No la contradigas ni la reinterpretes.**

Índices:

- `l`, `m` ∈ {0, …, n-1} → parches espaciales
- `i`, `j`, `k` ∈ {0, …, c-1} → serotipos / cepas

Compartimentos (todos son proporciones respecto de la población del parche):

| Símbolo TEX | Variable Python | Forma | Significado |
|---|---|---|---|
| $S_l$ | `S` | `(n,)` | Humanos susceptibles a todos los serotipos |
| $I_{li}$ | `I` | `(n, c)` | Infección **primaria** por serotipo `i` |
| $Z_{li}$ | `Z` | `(n, c)` | Recuperados de infección primaria por `i`, susceptibles al resto |
| $Y_{lij}$ | `Y` | `(n, c, c)` | Infección **secundaria**: primaria por `i`, secundaria por `j` |
| $V_{li}$ | `V` | `(n, c)` | Mosquitos infectados por serotipo `i` |

**Convención de `Y`: el primer índice de cepa es la infección PRIMARIA, el segundo es
la SECUNDARIA.** Por lo tanto, para contar humanos que infectan con el serotipo `i`
hay que sumar `I[:, i] + Y[:, :, i].sum(axis=1)`, es decir, sumar sobre el **eje 1**.
Esto es lo que hace `np.sum(Y, axis=1)` en el código actual y es correcto.

> Nota: el paper usa la convención opuesta ($Y_{ki}$ = primaria por $k$, secundaria
> por $i$, sumando sobre el primer índice). El TEX del proyecto es ambiguo en este
> punto. **El código es la referencia; el TEX se corrige para coincidir con él,
> nunca al revés.**

Compartimento no rastreado: los recuperados de **ambas** infecciones no tienen
variable propia. Salen del sistema por el término `-γ·Y` y no vuelven. Esto es
intencional (no hay reinfección triple) y es relevante para el test de conservación.

Matrices de acoplamiento:

| Símbolo TEX | Variable Python | Forma | Significado |
|---|---|---|---|
| $\lambda_{lim}$ | `lam` | `(n, c, n)` | Fuerza de infección sobre humanos de `l` ejercida por mosquitos de `m`, cepa `i` |
| $\delta_{lim}$ | `delta` | `(n, c, n)` | Infección de mosquitos de `l` por humanos de `m`, cepa `i` |
| $\sigma_{ij}$ | `sigma` | `(c, c)` | Efecto cruzado: primaria `i` → secundaria `j` |

Contracciones (respetar exactamente estos patrones de `einsum`):

```
TasaCont[l,i]   = Σ_m lam[l,i,m] · V[m,i]          'lim,mi->li'
TasaCont2[l,i,j] = Z[l,i] · TasaCont[l,j] · σ[i,j] · mask[i,j]   'li,lj,ij->lij'
dV[l,i]         = (1-Σ_v V[l,v]) · Σ_m delta[l,i,m] · H[m,i] - ν·V[l,i]
```

Semántica de `sigma`: `σ_ij = 1` → sin efecto cruzado; `0 ≤ σ_ij < 1` → protección
parcial o total; `σ_ij > 1` → **enhancement** (ADE, la infección primaria facilita la
secundaria). La diagonal se anula con `mask = 1 - I` porque no hay reinfección por
la misma cepa. **El valor por defecto actual es `σ = 1.5` fuera de la diagonal, o
sea régimen de enhancement.** Tenerlo presente al interpretar cualquier resultado.

---

## 3. Reglas duras

1. **No modifiques la matemática del modelo sin que `TAREA.md` lo pida
   explícitamente.** Refactorizar ≠ cambiar semántica. Si un cambio altera los
   valores numéricos de una corrida determinista, no es un refactor.
2. **Cualquier cambio a un `einsum` o a la construcción de `lam`/`delta` exige correr
   la suite de tests antes de dar la tarea por terminada.** Estos son los puntos
   donde un error es silencioso: la simulación sigue corriendo y produce un mapa de
   calor plausible pero incorrecto.
3. **No agregues dependencias** más allá de `numpy`, `scipy`, `matplotlib` (+ `pytest`
   para tests) sin preguntar primero.
4. **No toques el documento LaTeX** salvo que `TAREA.md` lo indique.
5. **No publiques, no hagas push, no crees repositorios remotos.** El autor lo hace
   a mano.
6. **No inventes datos epidemiológicos.** Este modelo todavía no está calibrado
   contra series reales. Si un README o docstring necesita números, usá los del
   paper y citalo.
7. Si encontrás un problema fuera del alcance de la sesión, **anotalo en
   `HALLAZGOS.md` y seguí**; no lo arregles por tu cuenta.

---

## 4. Problemas conocidos (NO arreglar salvo pedido explícito)

Están documentados a propósito. Varios son material para el TEX.

1. **El "cortafuegos" (Escenario 2) no aísla.** La pared central se inicializa con
   `Z0[idx,0] = 1.0` (recuperados de la cepa 0), pero con `σ = 1.5` esos nodos son
   **más** susceptibles que un nodo virgen a las cepas restantes. Con `c ≥ 2` la
   pared acelera el paso en vez de bloquearlo. Solo funciona como barrera con `c = 1`.
   Esto no es un bug a tapar: es un resultado del modelo (una barrera de inmunidad
   monoserotípica no es barrera bajo ADE) y va al TEX contrastando `σ<1` vs `σ>1`.
2. **Nombres de índices engañosos en la sección de visualización.** `idx_S` es en
   realidad el inicio de `I`, e `idx_Z` el inicio de `Y`. Los slices dan bien por
   cancelación de offsets, pero es frágil. Debe reemplazarse por los mismos `cuts`
   que usa `MapeoInv`.
3. **Acoplamiento denso.** `lam` y `delta` son `(n, c, n)` densos con solo ~5 vecinos
   no nulos por nodo. Para `n = 2500` son cientos de MB y cada evaluación del lado
   derecho cuesta O(n²c) en lugar de O(nc).
4. **Efecto de borde no normalizado.** La fuerza entrante total es
   `β(1 + k·coupling)` con `k` = 4 en el interior, 3 en bordes, 2 en esquinas. El
   frente de onda se deforma al llegar al borde por una razón artificial.
5. **Ambigüedad en la interpretación de `delta`.** Mosquitos del parche `l` se
   infectan de humanos de `m`, pero los mosquitos son sedentarios. Solo tiene sentido
   como "humanos de `m` visitando `l`" (marco de tiempos de residencia). El TEX
   arrastra la misma ambigüedad.
6. **`lam` y `delta` son formalmente independientes** aunque describen el mismo patrón
   de encuentro. Esto genera parámetros no identificables.

---

## 5. Convenciones de trabajo

- **Idioma:** comentarios, docstrings y README en español. Identificadores de código
  como están (mezcla de español e inglés) — no los renombres masivamente.
- **Estilo:** PEP 8, líneas ≤ 100 caracteres, type hints en funciones públicas.
- **Git:** una rama por sesión (`sesion-a-tests`, `sesion-b-refactor`, …). Commits
  chicos y atómicos, mensaje imperativo en español. El primer commit del repo debe
  ser el código original **sin tocar**, para poder revertir.
- **Tests:** `pytest`, en `tests/`, rápidos (la suite entera < 30 s). Nada de tests
  que dependan de mostrar una ventana de matplotlib.
- **Determinismo:** cualquier test que use aleatoriedad fija la semilla.

---

## 6. Flujo por sesiones

- **Sesión A — verificación.** Construir la suite de tests. Extracción mínima del
  modelo a un módulo importable, sin cambios semánticos. *Precede a todo lo demás.*
- **Sesión B — refactor e higiene.** CLI, modularización, acoplamiento disperso,
  README, GIF, licencia. **Los tests de la Sesión A deben pasar sin modificarse.**
- **Sesión C — LaTeX.** Subsección de condiciones iniciales como diseño de
  experimentos numéricos; tabla de correspondencia símbolo ↔ variable ↔ forma.

Entrá siempre en modo plan (Shift+Tab) antes de ejecutar.
