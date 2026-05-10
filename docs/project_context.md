# Contexto General Del Repositorio

Este documento resume el estado actual del repositorio `Jenovo22/multimodal-financial-forecasting` y sirve como guia de onboarding tecnico. Describe el proposito del proyecto, la organizacion de datos, la arquitectura del sistema, el diseno de la red FINN, el estado de API/dashboard, las instrucciones de instalacion/ejecucion y las tareas mas importantes a partir de este punto.

Fecha de referencia local: 2026-05-08.

## 1. Resumen Ejecutivo

El repositorio implementa un MVP de investigacion para valorar opciones de SPY usando una arquitectura hibrida:

- Un baseline analitico Black-Scholes-Merton con dividendos opcionales y Greeks.
- Un detector de regimenes de mercado tipo HMM para estimar contexto de volatilidad.
- Una red FINN, Finance-Informed Neural Network, entrenada para producir un valor razonable de opciones.
- Un pipeline de scoring que combina baseline, regimenes y FINN para generar `fair_value`, `edge`, Greeks y senal.
- Una API FastAPI para scoring por fila o batch.
- Un dashboard HTML estatico para inspeccion visual de predicciones, strikes, expiraciones, confianza y recomendacion.
- Tests automatizados, CI, plantillas de issues/PR y documentacion colaborativa.

El objetivo no es todavia ejecutar trading real automaticamente. El objetivo actual es construir una base robusta de investigacion, validacion y decision asistida. Las salidas deben tratarse como senales analiticas, no como asesoramiento financiero.

## 2. Estado Actual Del Repositorio

### 2.1 Estado Git Y Ramas

El repositorio esta organizado alrededor de `main` y tres ramas principales de experimentacion:

- `experiment/model-architecture`: cambios sobre FINN, arquitectura de red, regularizacion, PDE loss y experimentos de modelo.
- `experiment/data-evaluation`: cambios sobre data, snapshots, split temporal, manifests, evaluacion y backtesting.
- `experiment/dashboard-api`: cambios sobre dashboard, API, scoring, senales, recomendaciones y UX.

Flujo recomendado:

```powershell
git checkout experiment/data-evaluation
git checkout -b feature/timestamp-split-dataset
```

Luego se abre PR hacia la rama base correspondiente o hacia `main`, segun el acuerdo del equipo.

### 2.2 Estado Funcional Del MVP

Actualmente existe un flujo funcional de extremo a extremo:

```text
Datos de mercado / opciones
        |
        v
Normalizacion y dataset canonico
        |
        v
Regimen HMM + Black-Scholes-Merton
        |
        v
FINN residual anclada a BSM
        |
        v
Scoring de opciones
        |
        +--> API FastAPI
        |
        +--> Dashboard HTML local
```

Los checks locales al momento de crear esta documentacion estan verdes:

```text
76 tests passing
```

### 2.3 Resultado De Entrenamiento Local Disponible

El ultimo entrenamiento local registrado en `reports/finn_training_metrics.json` usa:

- `split_strategy`: `expiration`
- `prediction_mode`: `bsm_residual`
- `hidden_dims`: `[32, 32]` como default validado en la rama de arquitectura
- `activation`: `silu`
- `lambda_boundary`: `0.1`
- `lambda_pde`: `0.0`
- `lambda_arbitrage`: `1.0`
- `early_stopping_patience`: `75`
- `best_epoch`: `224`

Split actual:

| Split | Filas | Expiraciones | Timestamp |
| --- | ---: | --- | --- |
| train | 126 | `2026-05-29`, `2026-06-05` | `2026-05-05T00:00:00Z` |
| validation | 50 | `2026-06-12` | `2026-05-05T00:00:00Z` |
| test | 62 | `2026-06-18` | `2026-05-05T00:00:00Z` |

Metricas principales:

| Modelo | Split | MAE | RMSE | BSM MAE |
| --- | --- | ---: | ---: | ---: |
| selection FINN | test | `0.9240` | `1.2665` | `1.4930` |
| final FINN | test | `0.2355` | `0.2632` | `1.4930` |

Interpretacion:

- El modelo FINN residual mejora claramente contra BSM en este test por expiracion.
- La validacion todavia no es temporal real porque solo hay un snapshot de opciones: `2026-05-05`.
- La siguiente validacion robusta debe hacerse con multiples snapshots diarios y `--split-strategy timestamp`.

## 3. Organizacion Actual De Archivos

### 3.1 Estructura Principal

```text
src/                  Codigo principal del paquete Python
scripts/              Workflows reproducibles por CLI
tests/                Tests unitarios/integracion
docs/                 Documentacion operativa y colaborativa
markdown/             Especificaciones MVP y schema legacy
Data/                 Datos historicos/locales
artifacts/            Checkpoints locales, ignorados por Git
reports/              Reportes locales, ignorados por Git
notebooks/            Notebooks exploratorios/legacy
.github/              CI, plantillas de PR e issues
```

### 3.2 Modulos Principales En `src/`

```text
src/contracts.py
src/baseline/
src/data/
src/models/hmm/
src/models/finn/
src/pipeline/
src/api/
src/visualization/
src/backtesting/
```

Descripcion:

- `src/contracts.py`: contratos tipados compartidos, como `DatasetRow`, `BaselineInput`, `HMMInput`, `FINNInput`, `SystemOutput`.
- `src/baseline/`: Black-Scholes-Merton y Greeks.
- `src/data/`: normalizacion de fuentes de opciones, construccion de dataset canonico, union con mercado y regimenes.
- `src/models/hmm/`: HMM real entrenable y detector respaldado por tabla de regimenes.
- `src/models/finn/`: FINN, funciones de perdida, modelo fallback analitico y utilidades de entrenamiento/evaluacion.
- `src/pipeline/`: orquestacion HMM + FINN + BSM para scoring.
- `src/api/`: FastAPI.
- `src/visualization/`: generador del dashboard HTML.
- `src/backtesting/`: scaffold para backtesting.

### 3.3 Scripts Principales

| Script | Proposito |
| --- | --- |
| `scripts/setup_environment.ps1` | Crear entorno, instalar extras y correr tests |
| `scripts/run_checks.ps1` | Compilar y correr test suite |
| `scripts/build_spy_regime_table.py` | Construir tabla de regimenes SPY |
| `scripts/download_option_chain.py` | Descargar cadena de opciones y opcionalmente construir dataset |
| `scripts/update_spy_options.ps1` | Automatizar actualizacion de opciones SPY |
| `scripts/train_finn_model.py` | Entrenar FINN con split, checkpoint y metricas |
| `scripts/score_option_dataset.py` | Puntuar dataset de opciones con FINN/API pipeline |
| `scripts/run_api.py` | Levantar API local |
| `scripts/build_prediction_dashboard.py` | Generar dashboard HTML |

## 4. Organizacion Actual De La Data

### 4.1 Politica De Datos

El repositorio contiene algunos datos historicos ya versionados. Sin embargo, por politica actual:

- Los datos generados nuevos no deben commitearse.
- Los checkpoints `.pt` no deben commitearse.
- Los reportes generados no deben commitearse.
- Se prefiere documentar comandos reproducibles en vez de subir blobs.

Archivos/carpetas generadas ignoradas por Git:

```text
Data/
artifacts/
reports/
```

Nota importante: algunos archivos dentro de `Data/` ya estaban versionados historicamente. No deben borrarse casualmente; si se decide migrar datos fuera de Git, debe hacerse en una PR dedicada de data migration.

### 4.2 Dataset De Mercado Principal

Archivo local:

```text
Data/final/master_dataset_2010_present_long.csv
```

Estado local:

- Filas: `19,695`
- Columnas: `31`
- Fechas unicas: `3,939`
- Activos: `GLD`, `GOLD_FUTURES`, `SP500_INDEX`, `SPY`, `VOO`

Columnas principales:

```text
date
asset
ticker
open
high
low
close
adj_close
volume
price_used
return_1d
log_return_1d
return_5d
return_20d
volatility_5d
volatility_20d
volatility_60d
volatility_5d_ann
volatility_20d_ann
volatility_60d_ann
vix_fred
dgs10
dgs2
term_spread_10y_2y
fedfunds
cpi
unemployment
dollar_index
```

Uso en el proyecto:

- `DatasetBuilder` filtra `asset == SPY`.
- Usa `price_used` como `S`.
- Usa `return_1d`, `return_5d`, `volatility_20d_ann`, `vix_fred` y `dgs2` como contexto.
- Convierte `dgs2` a tasa `r` usando escala `0.01`.
- Usa `vix_fred` como proxy de volatilidad implicita de mercado, tambien con escala `0.01`.

### 4.3 Dataset Wide

Archivo:

```text
Data/final/master_dataset_2010_present_wide.csv
```

Estado local:

- Filas: `3,939`
- Columnas: `109`

Este formato agrupa los activos en columnas. Es util para exploracion y modelado macro/multiactivo, pero el pipeline actual usa principalmente el formato long.

### 4.4 Raw Options

Archivo local actual:

```text
Data/raw/options/SPY_options_2026-05-05.csv
```

Estado local:

- Filas: `1,435`
- Columnas: `20`
- Underlying: `SPY`
- Tipos: `call`, `put`
- Expiraciones: `2026-05-29`, `2026-06-05`, `2026-06-12`, `2026-06-18`

Columnas principales:

```text
contractSymbol
lastTradeDate
strike
lastPrice
bid
ask
change
percentChange
volume
openInterest
impliedVolatility
inTheMoney
contractSize
currency
underlying_symbol
option_type
expiration_date
quote_timestamp
downloaded_at
source
```

El raw contiene calls y puts, pero el dataset MVP actual filtra a calls.

### 4.5 Dataset Canonico De Opciones

Archivo local:

```text
Data/processed/option_training_dataset.csv
```

Estado local:

- Filas: `238`
- Columnas: `33`
- Timestamp unico: `2026-05-05T00:00:00Z`
- Underlying: `SPY`
- Option type: `call`
- Expiraciones: 4

Filtros actuales del MVP:

- `underlying_symbol == SPY`
- `option_type == call`
- `T` entre `20/365` y `45/365`
- `moneyness` entre `0.90` y `1.10`
- `volume >= 10` si existe
- `relative_spread <= 0.25` si existe

Columnas derivadas:

- `spread = ask - bid`
- `relative_spread = spread / mid_price`
- `moneyness = S / K`
- `log_moneyness = log(moneyness)`

### 4.6 Tabla De Regimenes HMM

Archivo local:

```text
Data/processed/spy_regime_features.csv
```

Estado local:

- Filas: `3,939`
- Columnas: `13`
- Underlying: `SPY`
- Regimenes: `stable_low_volatility`, `stress_high_volatility`, `transition_uncertainty`

Columnas:

```text
timestamp
underlying_symbol
S
return_1d
return_5d
realized_volatility
implied_volatility_proxy
r
regime_label
sigma_regime
regime_probability_stable_low_volatility
regime_probability_stress_high_volatility
regime_probability_transition_uncertainty
```

Esta tabla es usada por `RegimeTableDetector` en runtime. El pipeline no entrena el HMM cada vez; consume una tabla precomputada.

### 4.7 Datasets De Entrenamiento FINN

Archivos locales:

```text
Data/processed/finn_split_dataset.csv
Data/processed/finn_split_predictions.csv
```

`finn_split_dataset.csv`:

- Filas: `238`
- Incluye todas las columnas del dataset canonico.
- Agrega columnas de regimen.
- Agrega columna `split` con `train`, `validation`, `test`.

`finn_split_predictions.csv`:

- Filas: `300`
- Incluye predicciones por fila para modelo `selection` y modelo `final`.
- Incluye errores, precio baseline y metricas por fila.

### 4.8 Dataset De Scoring

Archivo local:

```text
Data/processed/scored_options.csv
```

Estado local:

- Filas: `238`
- Columnas: `17`
- Senales: `BUY`, `SELL`
- Regimen local actual en scoring: `transition_uncertainty`

Columnas:

```text
timestamp
underlying_symbol
option_symbol
market_price
bs_price
fair_value
edge
delta
gamma
regime_label
regime_probabilities
signal
transaction_cost_estimate
safety_margin
regime_probability_stable_low_volatility
regime_probability_stress_high_volatility
regime_probability_transition_uncertainty
```

El `edge` se calcula como:

```text
edge = fair_value - market_reference_price
```

La senal se calcula segun:

```text
threshold = transaction_cost_estimate + safety_margin

if edge > threshold: BUY
if edge < -threshold: SELL
else: HOLD
```

Actualmente `transaction_cost_estimate` y `safety_margin` estan en `0.0` por defecto, por lo que las senales son sensibles a edges pequenos.

## 5. Arquitectura General Del Sistema

### 5.1 Objetivo Funcional

El sistema intenta responder:

> Dada una opcion de SPY y el contexto de mercado, cual es su valor razonable estimado, cuanto difiere del precio observado y que senal analitica produce esa diferencia?

Para eso combina:

- Precio observado de mercado.
- Precio teorico BSM.
- Regimen de mercado HMM.
- Red FINN con restricciones financieras.
- Umbral de decision configurable por costos y margen de seguridad.

### 5.2 Flujo De Datos Completo

```text
Market dataset long
        |
        v
DatasetBuilder.build_market_context_frame()
        |
        +--------------------------+
        |                          |
        v                          v
HMM regime table             Option raw source
        |                          |
        |                          v
        |                   OptionSourceNormalizer
        |                          |
        |                          v
        |              DatasetBuilder.build_option_training_dataset()
        |                          |
        +------------ join --------+
                       |
                       v
              FINN training frame
                       |
                       v
          split train / validation / test
                       |
                       v
              FINNPricingModel.fit()
                       |
                       v
              artifacts/finn_model.pt
                       |
                       v
              score_option_dataset.py
                       |
                       v
              scored_options.csv
                       |
          +------------+------------+
          |                         |
          v                         v
      FastAPI                  HTML dashboard
```

### 5.3 Contratos Centrales

El archivo `src/contracts.py` define los objetos que conectan modulos.

`DatasetRow` representa una opcion ya enriquecida:

- `timestamp`
- `underlying_symbol`
- `option_symbol`
- `option_type`
- `S`
- `K`
- `T`
- `r`
- `market_price`
- `implied_volatility`
- `bid`, `ask`, `mid_price`
- `volume`, `open_interest`
- `realized_volatility`
- `return_1d`, `return_5d`
- `sentiment_score`, `text_embedding`, `event_count`
- `dividend_yield`

`BaselineInput`:

- `S`, `K`, `T`, `r`, `sigma`, `option_type`, `dividend_yield`

`HMMInput`:

- `timestamp`
- `HMMFeatures`: retornos, volatilidad realizada, volatilidad implicita, volumen, sentimiento, embedding

`HMMOutput`:

- `regime_label`
- `regime_probabilities`
- `sigma_regime`

`FINNInput`:

- `S`, `K`, `T`, `r`
- `sigma_regime`
- `option_type`
- `dividend_yield`
- `regime_probabilities`
- `z_t`

`SystemOutput`:

- `market_price`
- `bs_price`
- `fair_value`
- `edge`
- `delta`
- `gamma`
- `regime_label`
- `regime_probabilities`
- `signal`
- costos/margen

### 5.4 Baseline Black-Scholes-Merton

Modulo:

```text
src/baseline/black_scholes.py
```

Implementa Black-Scholes-Merton con dividend yield opcional:

- Normal CDF y PDF con `math.erf`.
- Calculo de `d1`, `d2`.
- Descuento por tasa libre de riesgo `exp(-rT)`.
- Descuento por dividendos `exp(-qT)`.
- Precio call/put.
- Greeks:
  - delta
  - gamma
  - vega
  - theta
  - rho

Validaciones:

- `S`, `K`, `T`, `sigma` estrictamente positivos.
- `option_type` en `call|put`.
- `dividend_yield >= 0`.
- `r` puede ser negativo o positivo; no se fuerza positividad.

Rol dentro del sistema:

- Sirve como baseline teorico.
- Sirve como comparacion de metricas.
- Sirve como ancla del modo `bsm_residual` de FINN.

### 5.5 HMM De Regimenes

Modulos:

```text
src/models/hmm/regime_detector.py
src/models/hmm/regime_table_detector.py
```

`HMMRegimeDetector` entrena un Gaussian HMM con `hmmlearn` sobre features de mercado:

- `return_1d`
- `return_5d`
- `realized_volatility`
- `implied_volatility`
- `volume`

Configuracion por defecto:

- `n_states = 3`
- `covariance_type = diag`
- `n_iter = 200`
- `tol = 1e-3`
- `random_state = 42`
- transformacion de volumen `log1p`

Proceso interno:

1. Extrae matriz de features.
2. Aplica transformaciones, imputacion por mediana y estandarizacion.
3. Entrena Gaussian HMM.
4. Calcula probabilidad posterior por estado.
5. Estima una volatilidad representativa por estado.
6. Ordena estados por volatilidad y los mapea a:
   - baja volatilidad estable
   - alta volatilidad stress
   - transicion/incertidumbre

`RegimeTableDetector` es el componente usado en runtime. No entrena; lee una tabla precomputada y devuelve `HMMOutput`.

Ventajas de la tabla:

- Scoring mas rapido y reproducible.
- API no depende de entrenar HMM al iniciar.
- Permite usar `match_mode = exact` o `previous`.

## 6. Diseno De La Red FINN

Esta es la parte central del proyecto desde el punto de vista de modelado.

### 6.1 Proposito De FINN

FINN significa Finance-Informed Neural Network. En este repositorio la FINN busca estimar el precio razonable de una opcion usando:

- Variables financieras clasicas.
- Regimen de mercado estimado por HMM.
- Restricciones inspiradas en finanzas:
  - bounds de precio vanilla
  - no-arbitraje via Delta/Gamma
  - residual PDE Black-Scholes, disponible pero actualmente desactivado

La idea no es reemplazar BSM ciegamente. La version robusta actual aprende una correccion residual sobre BSM. Esto reduce sobreajuste cuando hay pocos datos.

### 6.2 Entrada Conceptual

Contrato publico:

```python
FINNInput(
    S,
    K,
    T,
    r,
    sigma_regime,
    option_type,
    dividend_yield,
    regime_probabilities,
    z_t,
)
```

Donde:

- `S`: precio spot del subyacente.
- `K`: strike.
- `T`: tiempo a expiracion en anos.
- `r`: tasa libre de riesgo.
- `sigma_regime`: volatilidad asociada al regimen HMM.
- `option_type`: `call` o `put`.
- `dividend_yield`: yield de dividendos.
- `regime_probabilities`: probabilidad de cada regimen.
- `z_t`: embedding/texto reservado para multimodalidad futura.

### 6.3 Feature Engineering Interno

El modelo no usa directamente todos los campos crudos. Construye una matriz de 12 features:

```text
1. S
2. K
3. log(S / K)
4. T
5. sqrt(T)
6. r
7. sigma_regime
8. option_type_sign
9. dividend_yield
10. probability_stable_low_volatility
11. probability_stress_high_volatility
12. probability_transition_uncertainty
```

Detalles:

- `option_type_sign = +1` para call y `-1` para put.
- `log(S/K)` captura moneyness de forma mas estable que solo `S` y `K`.
- `sqrt(T)` ayuda porque la difusion en Black-Scholes escala con raiz del tiempo.
- Las probabilidades de regimen permiten que la red no dependa solo de una etiqueta discreta.
- `z_t` aun esta reservado, pero no esta conectado a la matriz de features actual.

### 6.4 Normalizacion

Durante `fit()`:

1. Se construye la matriz de features.
2. Se calcula `feature_mean_`.
3. Se calcula `feature_std_`.
4. Se estandariza:

```text
standardized = (features - feature_mean_) / feature_std_
```

Para columnas constantes, se usa `_stable_feature_std` para evitar division por valores extremadamente pequenos. Si el std cae debajo de `feature_std_floor`, se reemplaza por `1.0`.

Esto es importante porque en una sola cadena de opciones algunas features pueden ser constantes o casi constantes, por ejemplo `r`, `S`, `timestamp` implicito o probabilidades de regimen.

### 6.5 Arquitectura MLP

Configuracion actual:

```python
FINNConfig(
    input_dim=12,
    hidden_dims=(32, 32),
    activation="silu",
    prediction_mode="bsm_residual",
    residual_scale=0.50,
    residual_anchor_floor=1.0,
)
```

Arquitectura:

```text
Input 12
  |
Linear 12 -> 32
  |
SiLU
  |
Linear 32 -> 32
  |
SiLU
  |
Linear 32 -> 1
  |
raw_output
```

La salida de la red no siempre es precio directamente. Depende de `prediction_mode`.

### 6.6 Modo `direct`

En modo directo:

```text
price = softplus(raw_output)
```

Ventaja:

- La red puede aprender toda la funcion de pricing desde cero.

Desventaja:

- Es mas propensa a sobreajustar con pocos datos.
- Puede alejarse demasiado de una forma financiera razonable.

Este modo existe para experimentos, no es el default recomendado ahora.

### 6.7 Modo `bsm_residual`

Este es el modo recomendado actual.

Primero se calcula un precio ancla con Black-Scholes usando `sigma_regime`:

```text
anchor = BSM(S, K, T, r, sigma_regime, q, option_type)
```

Luego la red aprende una correccion acotada:

```text
residual_scale_value = max(abs(anchor), residual_anchor_floor) * residual_scale
price = clamp(anchor + residual_scale_value * tanh(raw_output), min=0)
```

Con los valores actuales:

```text
residual_scale = 0.50
residual_anchor_floor = 1.0
```

Interpretacion:

- Si BSM ancla en `10.0`, la red puede corregir aproximadamente dentro de un rango escalado de `+-5.0`.
- Si BSM ancla cerca de `0`, el floor evita que la red quede sin capacidad residual.
- `tanh` evita correcciones explosivas.
- `clamp(min=0)` evita precios negativos.

Adicionalmente, la capa de salida se inicializa en cero cuando el modo es `bsm_residual`. Como `tanh(0) = 0`, el modelo empieza exactamente en el ancla BSM. Eso hace que el entrenamiento parta desde una solucion financiera razonable.

Esta decision fue clave para mejorar estabilidad con pocos datos.

### 6.8 Loss Function Total

Modulo:

```text
src/models/finn/pde_loss.py
```

La perdida total es:

```text
total_loss =
    lambda_data * data_loss
  + lambda_boundary * boundary_loss
  + lambda_pde * pde_loss
  + lambda_arbitrage * arbitrage_loss
```

Configuracion actual:

```text
lambda_data = 1.0
lambda_boundary = 0.1
lambda_pde = 0.0
lambda_arbitrage = 1.0
```

#### Data Loss

MSE entre precio predicho y target:

```text
data_loss = mse(prediction, target_price)
```

El target actual usa `mid_price` si esta disponible; si no, usa `market_price`.

#### Boundary Loss

Penaliza violaciones de bounds de opciones vanilla.

Para calls:

```text
lower = max(S * exp(-qT) - K * exp(-rT), 0)
upper = S * exp(-qT)
```

Para puts:

```text
lower = max(K * exp(-rT) - S * exp(-qT), 0)
upper = K * exp(-rT)
```

Si el precio predicho cae fuera de esos bounds, se penaliza.

#### PDE Residual

Implementa residual tipo Black-Scholes usando autograd:

```text
dV/dT - [0.5 * sigma^2 * S^2 * d2V/dS2 + (r-q) * S * dV/dS - rV]
```

Actualmente `lambda_pde = 0.0`, por lo que el termino se calcula para diagnostico pero no afecta el entrenamiento. Esto es intencional porque con pocos datos puede desestabilizar el aprendizaje.

#### Arbitrage Loss

Penaliza violaciones basicas de Delta/Gamma:

Calls:

```text
0 <= delta <= exp(-qT)
```

Puts:

```text
-exp(-qT) <= delta <= 0
```

Y penaliza:

```text
gamma < 0
```

El objetivo es evitar formas de precio incompatibles con restricciones basicas de no-arbitraje.

### 6.9 Greeks Aprendidas

En `predict()`, FINN calcula Greeks con autograd:

- `delta = dV/dS`
- `gamma = d2V/dS2`
- `vega = dV/dsigma_regime`
- `theta = -dV/dT`
- `pde_residual`

Esto significa que el modelo no solo devuelve precio, sino tambien sensibilidad local de la funcion aprendida.

### 6.10 Entrenamiento

Script:

```text
scripts/train_finn_model.py
```

Flujo:

1. Construye frame FINN desde option source, market source y regime source.
2. Divide en train/validation/test.
3. Entrena `selection_model` solo con train y valida con validation.
4. Usa early stopping y guarda mejor epoch.
5. Evalua selection model en train/validation/test.
6. Entrena `final_model` con `train_val` por defecto, usando el best epoch.
7. Evalua final model en test.
8. Guarda:
   - checkpoint selection
   - checkpoint final
   - split dataset
   - split predictions
   - JSON de metricas

Split strategies:

- `expiration`: recomendado con un solo timestamp de opciones.
- `timestamp`: recomendado cuando haya varios snapshots diarios.
- `random`: solo debugging/smoke.

### 6.11 Limitaciones Actuales De FINN

- Solo hay un snapshot de opciones, por lo que no existe validacion temporal real.
- El dataset canonico actual contiene solo calls.
- `z_t`/text embeddings estan reservados, no integrados en la red.
- PDE loss esta desactivado.
- Las senales no incorporan todavia costos reales, slippage ni liquidez de forma estricta.
- La mejora contra BSM debe validarse en multiples fechas futuras antes de interpretarse como robusta.

## 7. API

Modulo:

```text
src/api/app.py
scripts/run_api.py
```

### 7.1 Proposito

Exponer el pipeline de scoring como servicio local:

- Health check.
- Metadata del runtime.
- Scoring por fila.
- Scoring por batch.

### 7.2 Configuracion

`APIServerConfig`:

- `service_name = proyecto-tam-api`
- `service_version = 0.1.0`
- `regime_source_path = Data/processed/spy_regime_features.csv`
- `finn_checkpoint_path`
- `transaction_cost_estimate`
- `safety_margin`
- `use_mid_price_if_available`
- `regime_match_mode = exact|previous`
- `regime_max_staleness_days`
- `regime_underlying_symbol = SPY`

### 7.3 Endpoints

```text
GET  /health
GET  /metadata
POST /score/row
POST /score/batch
```

`/health` devuelve:

- estado `ok` o `degraded`
- si el pipeline esta listo
- ruta de regimen
- filas de regimen
- nombre del componente FINN cargado
- error de startup si existe

`/metadata` devuelve:

- configuracion del servicio
- modo de match de regimen
- costos/margen
- componentes cargados:
  - `BlackScholesPricer`
  - `RegimeTableDetector`
  - `FINNPricingModel` o fallback

`/score/row` recibe un `DatasetRowRequest` y devuelve `SystemOutputResponse`.

`/score/batch` recibe una lista de filas y devuelve lista de outputs.

### 7.4 Fallback Si No Hay Checkpoint

Si se levanta la API sin `--finn-checkpoint`, usa:

```text
RegimeAdjustedBlackScholesFINN
```

Ese componente calcula BSM con `sigma_regime`. Sirve para mantener el pipeline funcional sin modelo entrenado.

### 7.5 Comando API

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt
```

Si el puerto 8000 esta ocupado:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt --port 8001
```

## 8. Dashboard

Modulos:

```text
src/visualization/prediction_dashboard.py
scripts/build_prediction_dashboard.py
```

### 8.1 Proposito

El dashboard es un HTML estatico para interpretar predicciones y senales sin depender de un servidor frontend. Permite inspeccionar:

- Precio observado.
- Fair value FINN.
- Precio BSM.
- Last traded price.
- Strike.
- Expiracion.
- Edge.
- Senal.
- Confianza.
- Contexto historico del subyacente.
- Simulador simple de retorno.

### 8.2 Inputs

Input principal:

```text
Data/processed/scored_options.csv
```

Input de metadata:

```text
Data/processed/option_training_dataset.csv
```

Input de historia:

```text
Data/final/master_dataset_2010_present_long.csv
```

### 8.3 Decision Metrics

El dashboard agrega columnas como:

- `predicted_price`
- `expected_profit`
- `expected_return_pct`
- `dte_days`
- `confidence_score`
- `confidence_label`
- `opportunity_score`
- `recommendation`
- `decision_reason`

La confianza combina:

- fuerza del edge
- spread
- volumen
- open interest
- acuerdo entre FINN y BSM
- confianza del regimen

Ponderacion actual:

```text
0.30 edge_strength
0.20 spread_score
0.15 liquidity_score
0.20 agreement_score
0.15 regime_confidence
```

### 8.4 Recomendacion

El dashboard transforma senales en recomendacion legible:

- Si confianza < 45: `WATCH`
- Si signal BUY: `BUY`
- Si signal SELL: `SELL / AVOID LONG`
- Si no: `HOLD`

Advertencia:

- En opciones muy baratas, el retorno porcentual puede verse muy alto aunque el edge en dolares sea pequeno.
- La recomendacion no incorpora todavia un modelo completo de ejecucion, slippage, comisiones o riesgo de cola.

### 8.5 Comando Dashboard

```powershell
.\.venv\Scripts\python.exe scripts\build_prediction_dashboard.py `
  --scored-path Data\processed\scored_options.csv `
  --output reports\prediction_dashboard.html `
  --open
```

## 9. Instalacion Y Ejecucion Del Repositorio

### 9.1 Requisitos

- Python `>=3.11`
- PowerShell en Windows
- Git
- Opcional: acceso a internet para `yfinance`
- Opcional: PyTorch para entrenar FINN

Dependencias base en `pyproject.toml`:

- `fastapi`
- `hmmlearn`
- `pandas`
- `uvicorn`

Extras:

- `dev`: `pytest`, `httpx`
- `data`: `yfinance`
- `ml`: `torch`

### 9.2 Instalacion Recomendada En Windows

Desde la raiz del repo:

```powershell
.\scripts\setup_environment.ps1 -WithData -WithML
```

Esto:

1. Crea `.venv` si no existe.
2. Actualiza `pip`.
3. Instala PyTorch CPU si se pasa `-WithML`.
4. Instala el proyecto editable con extras.
5. Compila archivos Python.
6. Corre tests si no se desactiva.

Activar entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 9.3 Instalacion Manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

### 9.4 Checks Locales

```powershell
.\scripts\run_checks.ps1
```

Equivalente manual:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

### 9.5 Construir Tabla De Regimenes

```powershell
.\.venv\Scripts\python.exe scripts\build_spy_regime_table.py `
  --output Data\processed\spy_regime_features.csv
```

### 9.6 Descargar Opciones SPY

```powershell
.\.venv\Scripts\python.exe scripts\download_option_chain.py `
  --symbol SPY `
  --min-dte-days 20 `
  --max-dte-days 45 `
  --quote-timestamp latest-market-date `
  --build-dataset
```

Automatizado:

```powershell
.\scripts\update_spy_options.ps1
```

### 9.7 Entrenar FINN Actual

```powershell
.\.venv\Scripts\python.exe scripts\train_finn_model.py `
  --option-source Data\raw\options\SPY_options_2026-05-05.csv `
  --regime-source Data\processed\spy_regime_features.csv `
  --output artifacts\finn_model.pt `
  --selection-output artifacts\finn_model_selection.pt `
  --split-output Data\processed\finn_split_dataset.csv `
  --metrics-output reports\finn_training_metrics.json `
  --predictions-output Data\processed\finn_split_predictions.csv `
  --split-strategy expiration `
  --train-fraction 0.60 `
  --validation-fraction 0.20 `
  --final-train-split train_val `
  --epochs 500 `
  --batch-size 64 `
  --learning-rate 0.001 `
  --prediction-mode bsm_residual `
  --residual-scale 0.50 `
  --lambda-boundary 0.1 `
  --lambda-pde 0 `
  --lambda-arbitrage 1.0 `
  --early-stopping-patience 75
```

### 9.8 Scoring De Opciones

```powershell
.\.venv\Scripts\python.exe scripts\score_option_dataset.py `
  --option-source Data\raw\options\SPY_options_2026-05-05.csv `
  --regime-source Data\processed\spy_regime_features.csv `
  --finn-checkpoint artifacts\finn_model.pt `
  --output Data\processed\scored_options.csv
```

Con costos y margen:

```powershell
.\.venv\Scripts\python.exe scripts\score_option_dataset.py `
  --option-source Data\raw\options\SPY_options_2026-05-05.csv `
  --regime-source Data\processed\spy_regime_features.csv `
  --finn-checkpoint artifacts\finn_model.pt `
  --output Data\processed\scored_options.csv `
  --transaction-cost-estimate 0.03 `
  --safety-margin 0.05
```

### 9.9 Levantar API

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt
```

Probar health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 9.10 Generar Dashboard

```powershell
.\.venv\Scripts\python.exe scripts\build_prediction_dashboard.py `
  --scored-path Data\processed\scored_options.csv `
  --output reports\prediction_dashboard.html `
  --open
```

## 10. Colaboracion E Issues

### 10.1 Plantillas Disponibles

```text
.github/ISSUE_TEMPLATE/bug_report.md
.github/ISSUE_TEMPLATE/feature_request.md
.github/ISSUE_TEMPLATE/experiment.md
.github/pull_request_template.md
```

Uso:

- Bugs reproducibles van por `bug_report.md`.
- Nuevas capacidades van por `feature_request.md`.
- Experimentos de modelo van por `experiment.md`.

### 10.2 Flujo De Trabajo

1. Crear o seleccionar issue.
2. Crear rama desde la rama base correcta.
3. Implementar el cambio minimo util.
4. Correr checks locales.
5. Regenerar metricas/reportes si cambia modelo/dashboard.
6. Actualizar docs/changelog.
7. Abrir PR.
8. Revisar riesgos, supuestos de data y tests.
9. Merge solo con checks verdes.

### 10.3 Ramas De Experimentacion

- `experiment/model-architecture`
- `experiment/data-evaluation`
- `experiment/dashboard-api`

## 11. Tareas Mas Importantes

Esta seccion detalla las tareas principales por frente. Cada una deberia convertirse en issue con objetivo, alcance, definicion de terminado y comando de validacion.

### 11.1 Frente 1: Data And Evaluation Foundation

Objetivo: pasar de un MVP con un solo snapshot a un sistema evaluable en tiempo real/historico.

#### Tarea 1: Crear workflow diario de snapshots de opciones

Problema:

Actualmente hay un solo snapshot de opciones: `2026-05-05`. Eso impide validar si el modelo generaliza a fechas futuras.

Alcance:

- Descargar diariamente cadenas de opciones SPY.
- Guardar raw por fecha.
- Evitar sobrescribir snapshots existentes.
- Registrar metadata: fecha, simbolo, fuente, expiraciones, filtros.

Salida esperada:

```text
Data/raw/options/SPY_options_YYYY-MM-DD.csv
```

Definicion de terminado:

- Existen comandos reproducibles.
- Cada snapshot conserva su fecha.
- No se commitean archivos generados.
- Se documenta el proceso en `docs/data_policy.md` o `docs/training.md`.

Rama sugerida:

```text
experiment/data-evaluation -> feature/daily-option-snapshots
```

#### Tarea 2: Dataset combinado multi-dia

Problema:

El entrenamiento actual acepta una fuente de opciones. Para validacion temporal se necesita combinar multiples snapshots.

Alcance:

- Crear script para combinar snapshots raw o datasets canonicos.
- Evitar duplicados por `timestamp`, `underlying_symbol`, `option_symbol`.
- Validar columnas requeridas.
- Generar un manifest con row counts.

Definicion de terminado:

- Comando crea dataset combinado.
- Tests cubren combinacion y deduplicacion.
- `train_finn_model.py` puede consumir la salida.

#### Tarea 3: Split temporal real

Problema:

El split por expiracion no prueba si el modelo funciona en fechas futuras.

Alcance:

- Usar `--split-strategy timestamp`.
- Entrenar con fechas antiguas.
- Validar con fechas intermedias.
- Testear con fechas futuras no vistas.

Definicion de terminado:

- `reports/finn_training_metrics.json` reporta timestamps distintos por split.
- FINN se compara contra BSM.
- PR documenta periodos exactos y row counts.

#### Tarea 4: Evaluacion con liquidez, spread y costos

Problema:

Un edge positivo no significa trade ejecutable. Opciones baratas pueden inflar ROI porcentual.

Alcance:

- Agregar metricas de spread absoluto/relativo.
- Calcular edge despues de medio spread.
- Filtrar por volumen/open interest.
- Reportar metricas por bucket de liquidez.

Definicion de terminado:

- Las metricas aparecen en reportes.
- Dashboard puede mostrar edge ajustado.
- Tests cubren formulas basicas.

### 11.2 Frente 2: Model And Research Validation

Objetivo: mejorar el modelo solo si supera baselines bajo validacion honesta.

#### Tarea 1: Registro de experimentos FINN vs BSM

Problema:

Los experimentos deben ser reproducibles y comparables.

Alcance:

- Usar `.github/ISSUE_TEMPLATE/experiment.md`.
- Registrar comando, dataset, split, metricas.
- Comparar contra BSM siempre.

Definicion de terminado:

- Cada experimento tiene issue.
- Cada PR de modelo incluye MAE/RMSE contra BSM.
- Si no mejora bajo test honesto, no se promueve.

#### Tarea 2: Backtesting historico

Problema:

El modelo predice fair value, pero falta medir decisiones a traves del tiempo.

Alcance:

- Definir reglas de entrada/salida.
- Simular costo, spread, slippage.
- Medir PnL, hit rate, drawdown, turnover.
- Separar performance por regimen.

Definicion de terminado:

- Backtester corre sobre snapshots historicos.
- Salida produce reporte reproducible.
- No se confunde pricing error con rentabilidad.

#### Tarea 3: Metadata de version de modelo

Problema:

La API y reportes no exponen suficiente trazabilidad del modelo cargado.

Alcance:

- Guardar version, timestamp, config, dataset fingerprint y metricas en checkpoint.
- Mostrar metadata en `/metadata`.
- Incluir modo de prediccion y best epoch.

Definicion de terminado:

- Checkpoint incluye metadata.
- API expone metadata.
- Tests verifican compatibilidad hacia atras.

#### Tarea 4: Calibracion de PDE loss

Problema:

El termino PDE existe, pero esta desactivado porque puede desestabilizar con poca data.

Alcance:

- Probar `lambda_pde` en grid controlado.
- Medir MAE, RMSE, Greeks y PDE residual.
- Validar con timestamp split cuando exista.

Definicion de terminado:

- Experimento documenta si PDE ayuda o perjudica.
- No se cambia default sin evidencia.

### 11.3 Frente 3: Decision Product, Dashboard And API

Objetivo: hacer que la salida del sistema sea entendible, prudente y util para decision asistida.

#### Tarea 1: Senales conscientes de liquidez/costos

Problema:

Actualmente BUY/SELL depende solo de `edge` contra un threshold simple.

Alcance:

- Integrar spread, comision y safety margin.
- Requerir liquidez minima para senales fuertes.
- Separar `model_signal` de `actionable_signal`.

Definicion de terminado:

- Senales no accionables quedan como `WATCH` o `HOLD`.
- Tests cubren edge pequeno, spread alto y baja liquidez.
- Dashboard explica el motivo.

#### Tarea 2: Paneles de riesgo en dashboard

Problema:

El usuario necesita entender expiracion, strike, probabilidad, confianza y riesgo de irse a cero.

Alcance:

- Agregar panel de riesgo por DTE, delta y moneyness.
- Mostrar edge ajustado por spread.
- Mostrar liquidez y confianza.
- Mostrar advertencia para opciones muy baratas/OTM.

Definicion de terminado:

- Dashboard muestra riesgo de forma visual.
- Tests verifican columnas nuevas.
- PR incluye captura o descripcion del reporte generado.

#### Tarea 3: API hardening

Problema:

La API funciona, pero todavia es local/MVP.

Alcance:

- Exponer metadata de modelo.
- Mejorar errores de validacion.
- Agregar version de schema.
- Documentar payloads ejemplo.

Definicion de terminado:

- Tests API actualizados.
- `docs/api.md` actualizado.
- `/metadata` informa modelo y configuracion.

#### Tarea 4: Documentacion y onboarding continuo

Problema:

El proyecto crece rapido; sin docs vivas se vuelve dificil colaborar.

Alcance:

- Mantener `docs/project_context.md`.
- Mantener `docs/workflow.md`.
- Actualizar `CHANGELOG.md`.
- Mantener README como puerta de entrada.

Definicion de terminado:

- Cada cambio relevante deja docs actualizadas.
- PR template se completa.
- Checks verdes.

## 12. Prioridad Recomendada

Orden recomendado:

1. Consolidar data temporal:
   - snapshots diarios
   - dataset combinado
   - timestamp split

2. Reentrenar y validar con fechas futuras:
   - FINN residual vs BSM
   - metricas por liquidez
   - metricas por regimen

3. Ajustar senales:
   - costos
   - spread
   - liquidez
   - safety margin

4. Mejorar dashboard/API:
   - decision panels
   - metadata de modelo
   - explicabilidad de recomendacion

5. Backtesting:
   - reglas de entrada/salida
   - PnL
   - drawdown
   - robustez por regimen

## 13. Caveats Importantes

- El rendimiento local actual es prometedor, pero no prueba generalizacion temporal.
- Las opciones baratas pueden mostrar ROI porcentual alto y aun asi ser malas operaciones.
- El sistema aun no modela slippage de forma completa.
- El dashboard ayuda a interpretar, pero no reemplaza gestion de riesgo.
- La API es local/MVP, no esta lista para produccion con seguridad, auth y monitoreo.
- FINN aprende sobre un universo filtrado: SPY calls, DTE 20-45, moneyness 0.90-1.10.
- Antes de ampliar a puts, otros underlyings o multimodal real, conviene cerrar la validacion temporal.

## 14. Archivos De Referencia

Contexto general:

- `README.md`
- `docs/index.md`
- `docs/workflow.md`
- `docs/project_context.md`

Data:

- `docs/data_policy.md`
- `src/data/dataset_builder.py`
- `src/data/option_source.py`
- `src/data/options_downloader.py`
- `src/data/finn_dataset.py`

Modelo:

- `docs/training.md`
- `src/models/finn/finn_model.py`
- `src/models/finn/pde_loss.py`
- `src/models/finn/training.py`
- `src/models/hmm/regime_detector.py`
- `src/models/hmm/regime_table_detector.py`

API/dashboard:

- `docs/api.md`
- `docs/dashboard.md`
- `src/api/app.py`
- `src/visualization/prediction_dashboard.py`

Colaboracion:

- `CONTRIBUTING.md`
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/experiment.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`
