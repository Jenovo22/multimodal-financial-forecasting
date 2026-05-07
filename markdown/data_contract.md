# Data Contract

## Proyecto
Modelo Multimodal de Predicción y Valoración Financiera con HMM + FINN

## Versión
0.1

## Propósito

Este documento define el contrato de datos para el MVP. Su objetivo es establecer qué columnas debe contener el dataset principal, qué tipos de datos se esperan, qué campos son obligatorios y qué validaciones mínimas deben cumplirse antes de entrenar modelos.

La base de datos se asume como ya existente. Este contrato define la vista o tabla final que consumirán los modelos.

---

## 1. Dataset principal

Nombre sugerido:

```text
model_training_dataset
```

Granularidad:

```text
1 fila = 1 opción en 1 timestamp
```

Frecuencia inicial:

```text
diaria
```

Universo inicial:

```text
Subyacente: SPY
Derivado: opciones Call
Vencimiento objetivo: aproximadamente 30 días
Moneyness: ATM ± 10%
```

---

## 2. Columnas obligatorias

| Columna | Tipo | Descripción | Requerida |
|---|---|---|---|
| timestamp | datetime | Fecha y hora de observación | Sí |
| underlying_symbol | string | Símbolo del subyacente, inicialmente SPY | Sí |
| option_symbol | string | Identificador único de la opción | Sí |
| option_type | string | call o put | Sí |
| S | float | Precio del subyacente | Sí |
| K | float | Strike de la opción | Sí |
| T | float | Tiempo a vencimiento en años | Sí |
| r | float | Tasa libre de riesgo anualizada | Sí |
| market_price | float | Precio objetivo de mercado | Sí |
| implied_volatility | float | Volatilidad implícita anualizada | Sí |

---

## 3. Columnas recomendadas

| Columna | Tipo | Descripción | Requerida |
|---|---|---|---|
| bid | float | Mejor precio comprador | No |
| ask | float | Mejor precio vendedor | No |
| mid_price | float | Promedio entre bid y ask | No |
| volume | float | Volumen negociado de la opción | No |
| open_interest | float | Interés abierto | No |
| realized_volatility | float | Volatilidad realizada del subyacente | No |
| return_1d | float | Retorno diario del subyacente | No |
| return_5d | float | Retorno acumulado de 5 días | No |
| sentiment_score | float | Score agregado de sentimiento | No |
| text_embedding | array[float] | Embedding textual agregado de la ventana | No |
| event_count | int | Número de eventos/noticias asociadas a la ventana | No |
| dividend_yield | float | Dividend yield anualizado, si aplica | No |

---

## 4. Columnas derivadas

Estas columnas pueden construirse durante el procesamiento.

| Columna | Fórmula | Descripción |
|---|---|---|
| mid_price | (bid + ask) / 2 | Precio medio de mercado |
| spread | ask - bid | Spread absoluto |
| relative_spread | (ask - bid) / mid_price | Spread relativo |
| moneyness | S / K | Relación precio/strike |
| log_moneyness | log(S / K) | Moneyness logarítmico |
| time_to_maturity_years | days_to_expiration / 365 | Tiempo a vencimiento |
| target_price | mid_price o market_price | Variable objetivo |
| edge_bs | bs_price - market_price | Diferencia baseline vs mercado |
| edge_finn | fair_value - market_price | Diferencia FINN vs mercado |

---

## 5. Reglas de calidad de datos

### 5.1 Reglas obligatorias

1. `timestamp` no puede ser nulo.
2. `option_symbol` no puede ser nulo.
3. `S > 0`.
4. `K > 0`.
5. `T > 0`.
6. `r` debe estar en escala decimal anualizada.
7. `implied_volatility > 0`.
8. `market_price > 0`.
9. `option_type` debe pertenecer a:

```text
call
put
```

10. No puede existir información futura en features usadas para entrenamiento.

---

## 6. Convenciones de unidades

| Variable | Unidad esperada |
|---|---|
| S | dólares |
| K | dólares |
| T | años |
| r | decimal anualizado |
| implied_volatility | decimal anualizado |
| realized_volatility | decimal anualizado |
| market_price | dólares |
| bid | dólares |
| ask | dólares |
| volume | contratos |
| open_interest | contratos |

Ejemplos:

```text
r = 0.045 significa 4.5% anual
implied_volatility = 0.22 significa 22% anual
T = 30 / 365 significa 30 días a vencimiento
```

---

## 7. Splits de entrenamiento

La separación debe ser temporal, nunca aleatoria simple.

Sugerencia inicial:

```text
Train: 70% inicial del periodo histórico
Validation: siguiente 15%
Test: último 15%
```

Alternativa recomendada para validación financiera:

```text
walk-forward validation
```

Regla crítica:

```text
No usar datos posteriores a la fecha de predicción para construir features anteriores.
```

---

## 8. Filtros iniciales del MVP

Para el primer MVP, filtrar el dataset así:

```text
underlying_symbol == "SPY"
option_type == "call"
T entre 20/365 y 45/365
moneyness entre 0.90 y 1.10
market_price > 0
implied_volatility > 0
volume mínimo configurable
relative_spread máximo configurable
```

Valores sugeridos:

```text
min_volume = 10
max_relative_spread = 0.25
```

---

## 9. Datasets derivados por módulo

### 9.1 Dataset para Black-Scholes

Columnas mínimas:

```text
timestamp
option_symbol
S
K
T
r
implied_volatility
dividend_yield
option_type
market_price
```

### 9.2 Dataset para HMM

Primera versión:

```text
timestamp
return_1d
return_5d
realized_volatility
implied_volatility
volume
sentiment_score
```

Segunda versión:

```text
timestamp
z_t
realized_volatility
implied_volatility
sentiment_score
```

### 9.3 Dataset para FINN

Primera versión:

```text
timestamp
option_symbol
S
K
T
r
sigma_regime
dividend_yield
regime_label
regime_probability_stable_low_volatility
regime_probability_stress_high_volatility
regime_probability_transition_uncertainty
option_type
market_price
target_price
```

Segunda versión:

```text
timestamp
option_symbol
S
K
T
r
implied_volatility
regime_probabilities
option_type
market_price
```

Orden canonico de `regime_probabilities`:

```text
(
  stable_low_volatility,
  stress_high_volatility,
  transition_uncertainty
)
```

### 9.4 Dataset para backtesting

```text
timestamp
option_symbol
underlying_symbol
market_price
bid
ask
mid_price
fair_value
delta
gamma
regime_label
regime_probabilities
edge
signal
```

---

## 10. Target principal

Target estadístico inicial:

```text
target_price = market_price
```

Si existen bid y ask:

```text
target_price = mid_price
```

Target financiero posterior:

```text
future_option_return
future_pnl_after_costs
```

Para el MVP, el objetivo principal es pricing, no predicción directa de retorno.

---

## 11. Validaciones automáticas sugeridas

Crear un validador que revise:

```text
- columnas obligatorias presentes
- tipos correctos
- nulos en campos críticos
- valores negativos inválidos
- timestamps duplicados por option_symbol
- T positivo
- implied_volatility positiva
- bid <= ask
- mid_price consistente
- moneyness dentro del rango del MVP
- splits temporales sin solapamiento
```

---

## 12. Ejemplo de fila

```json
{
  "timestamp": "2026-04-01",
  "underlying_symbol": "SPY",
  "option_symbol": "SPY_20260501_520_C",
  "option_type": "call",
  "S": 518.25,
  "K": 520.0,
  "T": 0.08219,
  "r": 0.045,
  "market_price": 8.40,
  "bid": 8.35,
  "ask": 8.45,
  "mid_price": 8.40,
  "volume": 1240,
  "open_interest": 18320,
  "implied_volatility": 0.214,
  "realized_volatility": 0.187,
  "return_1d": 0.0031,
  "return_5d": 0.0124,
  "sentiment_score": 0.18,
  "event_count": 14
}
```

---

## 13. Notas de implementación

1. El campo `text_embedding` puede almacenarse como array, vector separado o referencia a una tabla de embeddings.
2. Para la primera versión, el sistema debe funcionar aunque no exista `text_embedding`.
3. La multimodalidad completa se integrará después de validar Black-Scholes, HMM básico y FINN base.
4. El dataset debe ser exportable a Parquet para entrenamiento eficiente.
5. El pipeline puede normalizar chains crudos de opciones desde esquemas comunes, por ejemplo con columnas como `contractSymbol`, `lastTradeDate`, `strike`, `lastPrice`, `bid`, `ask`, `openInterest` e `impliedVolatility`.
