# granger-causality

**Granger causality testing with vector autoregression (VAR), lag selection via AIC/BIC, and impulse response analysis.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

**Granger causality** is a statistical hypothesis test for determining whether one time series is useful in forecasting another. A time series X Granger-causes Y if past values of X contain statistically significant information about future values of Y, beyond what Y's own past values provide.

This is **not** true causality in the philosophical sense — it's about **predictive precedence**. However, it's a powerful and widely-used tool in:

- **Econometrics** — Does money supply Granger-cause inflation?
- **Neuroscience** — Does activity in brain region A predict activity in region B?
- **Climate science** — Do ocean temperatures predict atmospheric patterns?
- **Finance** — Do trading volumes predict price movements?

The test works by comparing two vector autoregressive (VAR) models:
- **Restricted**: Y(t) = α₀ + Σ αₖ Y(t-k) + ε₁
- **Unrestricted**: Y(t) = α₀ + Σ αₖ Y(t-k) + Σ βₖ X(t-k) + ε₂

If the unrestricted model is significantly better (F-test), X Granger-causes Y.

## Features

- **`VARModel`** — Vector autoregression fitting via OLS with forecasting and AIC/BIC
- **`Autocorrelation`** — ACF and cross-correlation function computation
- **`GrangerTest`** — Granger causality F-test between two series
- **`LagSelection`** — Optimal lag order selection via AIC or BIC
- **`ImpulseResponse`** — Impulse response function from VAR coefficients

## Installation

```toml
[dependencies]
granger-causality = "0.1.0"
```

## Quick Start

```rust
use granger_causality::*;

// Test if X Granger-causes Y
let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3 + 1.0).cos()).collect();

let (f_stat, p_value, significant) = GrangerTest::test(&x, &y, 2);
println!("F-statistic: {:.4}", f_stat);
println!("p-value: {:.4}", p_value);
println!("Significant: {}", significant);
```

## VAR Model Fitting

```rust
use granger_causality::*;

// Two time series
let y1: Vec<f64> = (0..100).map(|i| i as f64).collect();
let y2: Vec<f64> = (0..100).map(|i| 2.0 * i as f64).collect();

// Fit VAR(2) model
let model = VARModel::fit(&[y1, y2], 2);
println!("Lag order: {}", model.lag());
println!("RSS for equation 1: {:.4}", model.rss(0));
println!("RSS for equation 2: {:.4}", model.rss(1));

// Forecast one step ahead
let past = vec![vec![99.0, 98.0], vec![198.0, 196.0]]; // [var][lag_idx]
let forecast = model.forecast_one_step(&past);
println!("Forecast: {:?}", forecast);

// Information criteria
println!("AIC: {:.4}", model.aic());
println!("BIC: {:.4}", model.bic());
```

## Lag Selection

Choosing the right lag order is crucial. Too few lags miss dynamics; too many overfit.

```rust
use granger_causality::*;

let data = vec![y1, y2];

// Select by AIC (tends to pick more lags)
let best_aic = LagSelection::by_aic(&data, 10);
println!("Best lag (AIC): {}", best_aic);

// Select by BIC (tends to pick fewer lags — more parsimonious)
let best_bic = LagSelection::by_bic(&data, 10);
println!("Best lag (BIC): {}", best_bic);
```

## Autocorrelation

```rust
use granger_causality::*;

let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();

// ACF up to lag 10
let acf = Autocorrelation::acf(&data, 10);
for (lag, rho) in acf.iter().enumerate() {
    println!("ACF({}): {:.4}", lag, rho);
}

// Cross-correlation
let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1 + 0.5).cos()).collect();
let ccf = Autocorrelation::ccf_at_lag(&x, &y, 3);
println!("CCF at lag 3: {:.4}", ccf);
```

## Impulse Response

```rust
use granger_causality::*;

let model = VARModel::fit(&[y1, y2], 2);
let ir = ImpulseResponse::new(&model);

// Response of all variables to a shock in variable 0
let responses = ir.compute(0, 10);
for (t, resp) in responses.iter().enumerate() {
    println!("t={}: {:?}", t, resp);
}
```

## Methodology

### VAR Estimation
The VAR(p) model Y(t) = Σ AₖY(t-k) + ε is estimated equation-by-equation using ordinary least squares (OLS). Each equation is:
```
Yᵢ(t) = cᵢ + Σₖ Σⱼ aₖᵢⱼ Yⱼ(t-k) + εᵢ(t)
```

The OLS solution uses the normal equations with Gaussian elimination.

### Granger Test
The F-statistic is:
```
F = ((RSS_restricted - RSS_unrestricted) / p) / (RSS_unrestricted / (T - 2p - 1))
```

where p is the lag order and T is the number of observations.

### AIC / BIC
```
AIC = T · ln(RSS/T) + 2k
BIC = T · ln(RSS/T) + k · ln(T)
```

where k is the number of parameters per equation.

### Impulse Response
The IRF traces the dynamic effects of a unit shock through the VAR system:
```
Y(t+h) = Σₖ Aₖ Y(t+h-k)
```

starting from a unit impulse in one variable.

## API Reference

| Type | Key Methods | Description |
|------|-------------|-------------|
| `VARModel` | `fit`, `forecast_one_step`, `aic`, `bic` | VAR model estimation and analysis |
| `Autocorrelation` | `acf`, `acf_at_lag`, `ccf_at_lag` | Correlation functions |
| `GrangerTest` | `test` | Granger causality F-test |
| `LagSelection` | `by_aic`, `by_bic` | Optimal lag selection |
| `ImpulseResponse` | `compute` | IRF computation |

## Performance

- **VAR fitting**: O(T · p² · n²) via OLS normal equations
- **Granger test**: O(T · p²) per test
- **Lag selection**: O(max_lag × VAR cost)
- **Impulse response**: O(horizon × lag × n²)

## License

MIT License. See [LICENSE](LICENSE) for details.
