# Technical Spec MVP

## Proyecto
Modelo Multimodal de Predicción y Valoración Financiera con HMM + FINN

## Versión
0.1

## Estado
Documento inicial para orientar el desarrollo del MVP.

---

## 1. Objetivo del MVP

Construir una primera versión funcional de un sistema cuantitativo capaz de valorar opciones financieras usando una arquitectura híbrida compuesta por:

1. Baseline financiero Black-Scholes-Merton.
2. Detector de regímenes de mercado con Hidden Markov Models, HMM.
3. Motor de valoración con Finance-Informed Neural Networks, FINN.
4. Extracción de griegas mediante autodiferenciación.
5. Backtesting de señales basadas en diferencia entre valor justo y precio de mercado.

La base de datos multimodal se asume como ya disponible. Este MVP empieza desde la construcción del dataset de entrenamiento, los módulos de modelado y la integración progresiva de la arquitectura.

---

## 2. Universo inicial

### 2.1 Subyacente

MVP inicial:

```text
SPY
```

Justificación:

- Alta liquidez.
- Mayor facilidad de acceso operativo que SPX.
- Buen punto de partida para validar arquitectura.
- Permite migrar posteriormente a SPX para un enfoque más institucional.

### 2.2 Derivado inicial

```text
Opciones Call sobre SPY
```

Alcance inicial:

```text
Estilo: aproximación europea
Frecuencia: diaria
Vencimiento objetivo: aproximadamente 30 días
Moneyness: ATM ± 10%
```

Extensiones posteriores:

```text
Puts
Vencimientos 7D, 14D, 30D, 60D
SPX
Frecuencia intradía
Mayor rango de strikes
```

---

## 3. Supuestos financieros iniciales

1. El primer modelo usará opciones Call.
2. Se usará una aproximación europea para facilitar el uso de Black-Scholes-Merton como baseline y como restricción PDE.
3. El precio de mercado objetivo será preferiblemente el mid price:

```text
mid_price = (bid + ask) / 2
```

4. La tasa libre de riesgo se tomará desde la curva disponible en la base de datos.
5. La volatilidad inicial será la volatilidad implícita disponible o, si no existe, una estimación histórica.
6. No se ejecutarán órdenes reales durante el MVP.
7. Toda lógica de ejecución será primero validada con backtesting y luego con paper trading.

---

## 4. Arquitectura objetivo

```text
Base de datos multimodal
  -> Dataset Builder
  -> Baseline Black-Scholes
  -> HMM Regime Detector
  -> FINN Pricing Engine
  -> Greeks Engine
  -> Signal Engine
  -> Backtesting Engine
  -> Paper Trading Layer
```

Versión posterior con multimodalidad completa:

```text
Base de datos multimodal
  -> Dataset Builder
  -> Multimodal Encoder
  -> HMM Regime Detector
  -> Regime-Adjusted Volatility
  -> FINN Pricing Engine
  -> Greeks Engine
  -> Signal Engine
  -> Backtesting Engine
  -> Paper Trading Layer
```

---

## 5. Módulos del MVP

### 5.1 Dataset Builder

Responsable de construir el dataset final desde la base de datos existente.

Entrada esperada:

```text
Tablas o vistas con precios del subyacente, cadena de opciones, tasas, volatilidad, volumen, open interest y señales textuales.
```

Nota de implementación:

```text
El Dataset Builder puede normalizar fuentes crudas de cadenas de opciones cuando vengan con
campos comunes como contractSymbol, lastTradeDate, strike, lastPrice, bid, ask,
openInterest e impliedVolatility.
```

Salida:

```text
model_training_dataset
```

Granularidad:

```text
1 fila = 1 opción en 1 timestamp
```

---

### 5.2 Baseline Black-Scholes

Responsable de calcular precios y griegas clásicas.

Inputs:

```text
S
K
T
r
sigma
dividend_yield opcional
option_type
```

Outputs:

```text
bs_price
bs_delta
bs_gamma
bs_vega
bs_theta
bs_rho
```

Objetivo:

Servir como baseline financiero y control de calidad para la FINN.

---

### 5.3 HMM Regime Detector

Responsable de inferir regímenes latentes de mercado.

Inputs iniciales:

```text
return_1d
return_5d
realized_volatility
implied_volatility
volume
sentiment_score
```

Número inicial de estados:

```text
3
```

Etiquetas sugeridas:

```text
0: stable_low_volatility
1: stress_high_volatility
2: transition_uncertainty
```

Outputs:

```text
regime_label
regime_probabilities
sigma_regime
```

---

### 5.4 FINN Pricing Engine

Responsable de estimar el valor justo de la opción respetando restricciones financieras.

Inputs de la primera versión:

```text
S
K
T
r
sigma_regime
dividend_yield
option_type
```

Output principal:

```text
fair_value
```

Pérdida compuesta:

```text
L_total = lambda_data * L_data
        + lambda_boundary * L_boundary
        + lambda_pde * L_pde
```

Extensión posterior:

```text
L_total = lambda_data * L_data
        + lambda_boundary * L_boundary
        + lambda_pde * L_pde
        + lambda_arbitrage * L_arbitrage
```

---

### 5.5 Greeks Engine

Responsable de extraer sensibilidades.

Outputs obligatorios:

```text
delta
gamma
```

Outputs posteriores:

```text
vega
theta
rho
```

Método:

```text
Autodiferenciación sobre la salida de la FINN
```

---

### 5.6 Signal Engine

Responsable de convertir el fair value en una señal.

Regla inicial:

```text
edge = fair_value - market_mid_price
```

Señales:

```text
BUY  si edge > transaction_costs + safety_margin
SELL si edge < -transaction_costs - safety_margin
HOLD en caso contrario
```

---

### 5.7 Backtesting Engine

Responsable de evaluar la utilidad financiera.

Debe considerar:

```text
bid-ask spread
comisiones
slippage
liquidez mínima
walk-forward validation
control de exposición
```

Métricas mínimas:

```text
PnL acumulado
Sharpe
Sortino
max drawdown
win rate
profit factor
turnover
```

---

## 6. Orden recomendado de implementación

```text
1. data_contract.md
2. model_io_schema.json
3. src/baseline/black_scholes.py
4. src/baseline/greeks.py
5. src/data/dataset_builder.py
6. src/models/hmm/regime_detector.py
7. src/models/finn/finn_model.py
8. src/models/finn/pde_loss.py
9. src/pipeline/hmm_finn_pipeline.py
10. src/backtesting/backtester.py
```

---

## 7. Criterios de éxito del MVP

El MVP será considerado exitoso si:

1. Construye un dataset reproducible.
2. Calcula precios Black-Scholes correctamente.
3. Calcula griegas Black-Scholes correctamente.
4. Entrena un HMM con regímenes interpretables.
5. Entrena una FINN funcional.
6. Integra HMM + FINN.
7. Produce fair value, Delta, Gamma, edge y señal.
8. Ejecuta backtesting sin leakage temporal.
9. Genera métricas estadísticas y financieras.

---

## 8. Métricas estadísticas

```text
MAE
RMSE
MAPE
pricing_error_by_moneyness
pricing_error_by_maturity
pricing_error_by_regime
```

---

## 9. Métricas financieras

```text
cumulative_pnl
sharpe_ratio
sortino_ratio
max_drawdown
win_rate
profit_factor
turnover
average_exposure
```

---

## 10. Riesgos principales

### 10.1 Leakage temporal

Ninguna variable futura debe entrar en el entrenamiento. Esto aplica especialmente a volatilidad implícita, precios de opciones, eventos textuales y features agregadas.

### 10.2 Sobreajuste

El sistema debe validarse con separación temporal y walk-forward validation.

### 10.3 Regímenes no interpretables

El HMM debe validarse económicamente. No basta con que produzca clusters estadísticos.

### 10.4 Inestabilidad en la pérdida FINN

La pérdida PDE puede dominar la pérdida de datos o quedar irrelevante. Debe monitorearse el balance de lambdas.

### 10.5 Señales no ejecutables

Un edge estadístico puede desaparecer después de spread, slippage y comisiones.

---

## 11. Entregables inmediatos

```text
technical_spec_mvp.md
data_contract.md
model_io_schema.json
```

Después de estos tres archivos, el primer desarrollo de código será:

```text
src/baseline/black_scholes.py
src/baseline/greeks.py
```
