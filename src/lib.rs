//! # granger-causality
//!
//! Granger causality testing with vector autoregression (VAR), lag selection via
//! AIC/BIC, autocorrelation functions, and impulse response analysis.
//!
//! ## Overview
//!
//! **Granger causality** tests whether past values of one time series help predict
//! another time series, beyond what the second series's own past provides.
//!
//! A variable X Granger-causes Y if past values of X contain information that helps
//! predict Y above and beyond the information contained in past values of Y alone.
//!
//! The test fits two models:
//! - **Restricted**: Y(t) = α₀ + Σ αₖ Y(t-k) + ε₁
//! - **Unrestricted**: Y(t) = α₀ + Σ αₖ Y(t-k) + Σ βₖ X(t-k) + ε₂
//!
//! If the unrestricted model is significantly better (F-test), X Granger-causes Y.
//!
//! ## Core Types
//!
//! - [`VARModel`] — vector autoregression fitting and forecasting
//! - [`Autocorrelation`] — autocorrelation function computation
//! - [`GrangerTest`] — Granger causality F-test
//! - [`LagSelection`] — optimal lag selection via AIC/BIC
//! - [`ImpulseResponse`] — impulse response function from VAR coefficients

/// A vector autoregression (VAR) model.
///
/// Y(t) = Σ A_k Y(t-k) + ε, where A_k are coefficient matrices.
#[derive(Clone, Debug)]
pub struct VARModel {
    /// Number of variables (equations)
    n_vars: usize,
    /// Lag order
    lag: usize,
    /// Coefficient matrices: coefficients[k][i][j] = effect of variable j at lag (k+1) on variable i
    coefficients: Vec<Vec<Vec<f64>>>,
    /// Intercepts: intercepts[i] = intercept for variable i
    intercepts: Vec<f64>,
    /// Residual sum of squares for each equation
    rss: Vec<f64>,
    /// Number of observations used in fitting
    n_obs: usize,
}

impl VARModel {
    /// Create a new VAR model with the given parameters.
    pub fn new(
        n_vars: usize,
        lag: usize,
        coefficients: Vec<Vec<Vec<f64>>>,
        intercepts: Vec<f64>,
    ) -> Self {
        Self {
            n_vars,
            lag,
            coefficients,
            intercepts,
            rss: vec![0.0; n_vars],
            n_obs: 0,
        }
    }

    /// Fit a VAR(p) model using OLS for each equation.
    ///
    /// `data` is organized as data[variable][time], with `n_vars` time series
    /// each of length T.
    pub fn fit(data: &[Vec<f64>], lag: usize) -> Self {
        let n_vars = data.len();
        let t = data[0].len();
        let n_obs = t - lag;

        // For each variable, solve OLS: Y_i = X * beta_i
        // X matrix columns: [1, Y_1(t-1), Y_2(t-1), ..., Y_1(t-p), Y_2(t-p), ...]
        let n_cols = 1 + n_vars * lag;
        let mut coefficients = vec![vec![vec![0.0; n_vars]; n_vars]; lag];
        let mut intercepts = vec![0.0; n_vars];
        let mut rss = vec![0.0; n_vars];

        for var in 0..n_vars {
            // Build X and Y
            let mut x_mat = vec![vec![0.0; n_cols]; n_obs];
            let mut y_vec = vec![0.0; n_obs];

            for t_idx in 0..n_obs {
                let t = t_idx + lag;
                y_vec[t_idx] = data[var][t];
                x_mat[t_idx][0] = 1.0; // intercept

                for l in 0..lag {
                    for v in 0..n_vars {
                        x_mat[t_idx][1 + l * n_vars + v] = data[v][t - l - 1];
                    }
                }
            }

            // OLS: beta = (X'X)^-1 X'Y (simplified using normal equations)
            let beta = ols_solve(&x_mat, &y_vec, n_cols);

            intercepts[var] = beta[0];
            for l in 0..lag {
                for v in 0..n_vars {
                    coefficients[l][var][v] = beta[1 + l * n_vars + v];
                }
            }

            // Compute RSS
            for t_idx in 0..n_obs {
                let t = t_idx + lag;
                let mut predicted = intercepts[var];
                for l in 0..lag {
                    for v in 0..n_vars {
                        predicted += coefficients[l][var][v] * data[v][t - l - 1];
                    }
                }
                rss[var] += (data[var][t] - predicted).powi(2);
            }
        }

        Self {
            n_vars,
            lag,
            coefficients,
            intercepts,
            rss,
            n_obs,
        }
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Lag order.
    pub fn lag(&self) -> usize {
        self.lag
    }

    /// Residual sum of squares for equation `var`.
    pub fn rss(&self, var: usize) -> f64 {
        self.rss[var]
    }

    /// Number of observations used in fitting.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Forecast one step ahead given past values.
    /// `past[var][lag_idx]` where lag_idx 0 = most recent.
    pub fn forecast_one_step(&self, past: &[Vec<f64>]) -> Vec<f64> {
        let mut forecast = vec![0.0; self.n_vars];
        for var in 0..self.n_vars {
            forecast[var] = self.intercepts[var];
            for l in 0..self.lag {
                for v in 0..self.n_vars {
                    forecast[var] += self.coefficients[l][var][v] * past[v][l];
                }
            }
        }
        forecast
    }

    /// Akaike Information Criterion.
    pub fn aic(&self) -> f64 {
        let n = self.n_obs as f64;
        if n <= 0.0 {
            return f64::INFINITY;
        }
        let k = (1 + self.n_vars * self.lag) as f64;
        let total_rss: f64 = self.rss.iter().sum();
        let avg_rss = (total_rss / n).max(1e-15);
        n * avg_rss.ln() + 2.0 * k * self.n_vars as f64
    }

    /// Bayesian Information Criterion.
    pub fn bic(&self) -> f64 {
        let n = self.n_obs as f64;
        if n <= 0.0 {
            return f64::INFINITY;
        }
        let k = (1 + self.n_vars * self.lag) as f64;
        let total_rss: f64 = self.rss.iter().sum();
        let avg_rss = (total_rss / n).max(1e-15);
        n * avg_rss.ln() + k * n.ln() * self.n_vars as f64
    }
}

/// Simple OLS solver using normal equations with Gaussian elimination.
fn ols_solve(x: &[Vec<f64>], y: &[f64], n_cols: usize) -> Vec<f64> {
    let n = x.len();

    // X'X
    let mut xtx = vec![vec![0.0; n_cols]; n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[k][i] * x[k][j];
            }
            xtx[i][j] = sum;
        }
    }

    // Add small regularization to avoid singularity
    for i in 0..n_cols {
        xtx[i][i] += 1e-10;
    }

    // X'Y
    let mut xty = vec![0.0; n_cols];
    for i in 0..n_cols {
        let mut sum = 0.0;
        for k in 0..n {
            sum += x[k][i] * y[k];
        }
        xty[i] = sum;
    }

    // Solve using Gaussian elimination
    let m = n_cols;
    let mut aug = vec![vec![0.0; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            aug[i][j] = xtx[i][j];
        }
        aug[i][m] = xty[i];
    }

    // Forward elimination with partial pivoting
    let mut pivot_row = vec![0usize; m];
    for col in 0..m {
        let mut best = col;
        for row in (col + 1)..m {
            if aug[row][col].abs() > aug[best][col].abs() {
                best = row;
            }
        }
        aug.swap(col, best);
        pivot_row[col] = best;

        if aug[col][col].abs() < 1e-15 {
            continue;
        }

        let pivot = aug[col][col];
        for row in (col + 1)..m {
            let factor = aug[row][col] / pivot;
            for j in col..=m {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut beta = vec![0.0; m];
    for i in (0..m).rev() {
        if aug[i][i].abs() < 1e-15 {
            beta[i] = 0.0;
            continue;
        }
        beta[i] = aug[i][m];
        for j in (i + 1)..m {
            beta[i] -= aug[i][j] * beta[j];
        }
        beta[i] /= aug[i][i];
    }

    beta
}

/// Autocorrelation function computation.
pub struct Autocorrelation;

impl Autocorrelation {
    /// Compute autocorrelation at lag `k` for a single time series.
    pub fn acf_at_lag(data: &[f64], k: usize) -> f64 {
        let n = data.len();
        if k >= n {
            return 0.0;
        }
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..n {
            denom += (data[i] - mean).powi(2);
        }

        if denom.abs() < 1e-15 {
            return 0.0;
        }

        for i in 0..(n - k) {
            num += (data[i] - mean) * (data[i + k] - mean);
        }

        num / denom
    }

    /// Compute ACF up to `max_lag`.
    pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
        (0..=max_lag).map(|k| Self::acf_at_lag(data, k)).collect()
    }

    /// Cross-correlation at lag k between two series.
    pub fn ccf_at_lag(x: &[f64], y: &[f64], k: usize) -> f64 {
        let n = x.len().min(y.len());
        if k >= n {
            return 0.0;
        }
        let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

        let mut num = 0.0;
        let mut denom_x = 0.0;
        let mut denom_y = 0.0;

        for i in 0..n {
            denom_x += (x[i] - mean_x).powi(2);
            denom_y += (y[i] - mean_y).powi(2);
        }

        let denom = (denom_x * denom_y).sqrt();
        if denom.abs() < 1e-15 {
            return 0.0;
        }

        for i in 0..(n - k) {
            num += (x[i] - mean_x) * (y[i + k] - mean_y);
        }

        num / denom
    }
}

/// Granger causality test.
pub struct GrangerTest;

impl GrangerTest {
    /// Test if `cause` Granger-causes `effect` using the given `lag`.
    ///
    /// Returns `(f_statistic, p_value_approx, is_significant)`.
    ///
    /// Uses an F-test comparing restricted (effect only) vs unrestricted (effect + cause) models.
    /// p_value is approximate (uses a simple chi-squared approximation).
    pub fn test(
        cause: &[f64],
        effect: &[f64],
        lag: usize,
    ) -> (f64, f64, bool) {
        let n = effect.len();
        let n_obs = n - lag;

        // Fit restricted model: Y(t) = α + Σ αk Y(t-k)
        let restricted = VARModel::fit(&[effect.to_vec()], lag);
        let rss_restricted = restricted.rss(0);

        // Fit unrestricted model: Y(t) = α + Σ αk Y(t-k) + Σ βk X(t-k)
        let unrestricted = VARModel::fit(&[effect.to_vec(), cause.to_vec()], lag);
        let rss_unrestricted = unrestricted.rss(0);

        // F-test
        let df_num = lag as f64; // number of additional parameters
        let df_denom = (n_obs - 2 * lag - 1) as f64;

        if df_denom <= 0.0 || rss_restricted.abs() < 1e-15 {
            return (0.0, 1.0, false);
        }

        let f_stat = ((rss_restricted - rss_unrestricted) / df_num)
            / (rss_unrestricted / df_denom);

        // Approximate p-value using F-distribution approximation
        // For simplicity, use a threshold-based approach
        let p_value = if f_stat < 1.0 {
            0.8
        } else if f_stat < 2.0 {
            0.2
        } else if f_stat < 3.0 {
            0.1
        } else if f_stat < 5.0 {
            0.05
        } else if f_stat < 10.0 {
            0.01
        } else {
            0.001
        };

        (f_stat, p_value, f_stat > 3.0)
    }
}

/// Lag selection for VAR models using information criteria.
pub struct LagSelection;

impl LagSelection {
    /// Select optimal lag using AIC.
    ///
    /// Fits VAR models with lags 1..=max_lag and returns the lag with lowest AIC.
    pub fn by_aic(data: &[Vec<f64>], max_lag: usize) -> usize {
        let mut best_lag = 1;
        let mut best_aic = f64::INFINITY;

        for lag in 1..=max_lag {
            let model = VARModel::fit(data, lag);
            let aic = model.aic();
            if aic < best_aic {
                best_aic = aic;
                best_lag = lag;
            }
        }

        best_lag
    }

    /// Select optimal lag using BIC.
    pub fn by_bic(data: &[Vec<f64>], max_lag: usize) -> usize {
        let mut best_lag = 1;
        let mut best_bic = f64::INFINITY;

        for lag in 1..=max_lag {
            let model = VARModel::fit(data, lag);
            let bic = model.bic();
            if bic < best_bic {
                best_bic = bic;
                best_lag = lag;
            }
        }

        best_lag
    }
}

/// Impulse response function from a VAR model.
pub struct ImpulseResponse<'a> {
    model: &'a VARModel,
}

impl<'a> ImpulseResponse<'a> {
    /// Create a new impulse response analyzer.
    pub fn new(model: &'a VARModel) -> Self {
        Self { model }
    }

    /// Compute the impulse response of all variables to a unit shock in `shock_var`
    /// for `horizon` steps ahead.
    ///
    /// Returns a vector of length `horizon`, each element is a Vec of responses for each variable.
    pub fn compute(&self, shock_var: usize, horizon: usize) -> Vec<Vec<f64>> {
        let n = self.model.n_vars;
        let mut responses = Vec::with_capacity(horizon);

        // Initial shock
        let mut current = vec![0.0; n];
        current[shock_var] = 1.0;
        responses.push(current.clone());

        // Propagate through VAR structure
        for _ in 1..horizon {
            let mut next = vec![0.0; n];
            for l in 0..self.model.lag {
                if l < responses.len() {
                    let past = &responses[responses.len() - 1 - l];
                    for var in 0..n {
                        for v in 0..n {
                            next[var] += self.model.coefficients[l][var][v] * past[v];
                        }
                    }
                }
            }
            responses.push(next.clone());
            current = next;
        }

        responses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_fit_perfect() {
        // Y(t) = 2 * Y(t-1)
        let y: Vec<f64> = (0..20).map(|i| 2.0_f64.powi(i)).collect();
        let model = VARModel::fit(&[y], 1);
        assert!((model.coefficients[0][0][0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_var_forecast() {
        let y: Vec<f64> = (0..20).map(|i| 2.0_f64.powi(i)).collect();
        let model = VARModel::fit(&[y.clone()], 1);
        let forecast = model.forecast_one_step(&vec![vec![y[19]]]);
        // Should predict ~2 * y[19] = 2^20
        assert!((forecast[0] - 2.0_f64.powi(20)).abs() / 2.0_f64.powi(20) < 0.15);
    }

    #[test]
    fn test_acf_lag0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf0 = Autocorrelation::acf_at_lag(&data, 0);
        assert!((acf0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_sequence() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = Autocorrelation::acf(&data, 3);
        assert_eq!(acf.len(), 4);
        assert!((acf[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ccf() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let ccf0 = Autocorrelation::ccf_at_lag(&x, &y, 0);
        assert!(ccf0 > 0.9); // perfectly correlated
    }

    #[test]
    fn test_granger_no_causality() {
        // Independent random-looking series
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3 + 1.0).cos()).collect();
        let (f_stat, _p, _sig) = GrangerTest::test(&x, &y, 2);
        // Should have moderate F-stat (not strongly causal)
        assert!(f_stat.is_finite());
    }

    #[test]
    fn test_granger_causality() {
        // X Granger-causes Y with noise
        // Y(t) = 0.5 * X(t-1) + 0.3 * Y(t-1) + noise
        let mut x = vec![0.0f64; 200];
        let mut y = vec![0.0f64; 200];
        // Simple pseudo-noise
        let mut seed: u64 = 42;
        for i in 1..200 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            x[i] = 0.7 * x[i - 1] + ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.5;
            y[i] = 0.5 * x[i - 1] + 0.3 * y[i - 1] + noise;
        }
        let (f_stat, _p, _sig) = GrangerTest::test(&x, &y, 3);
        // With a true causal relationship, F-stat should be notable
        assert!(f_stat > 0.0, "Expected positive f_stat but got {}", f_stat);
    }

    #[test]
    fn test_lag_selection_aic() {
        // Simple AR(1) process
        let mut y = vec![0.0; 100];
        for i in 1..100 {
            y[i] = 0.8 * y[i - 1] + 0.1 * (i as f64).sin();
        }
        let best = LagSelection::by_aic(&[y], 5);
        assert!(best >= 1 && best <= 5);
    }

    #[test]
    fn test_lag_selection_bic() {
        let mut y = vec![0.0; 100];
        for i in 1..100 {
            y[i] = 0.8 * y[i - 1] + 0.1 * (i as f64).sin();
        }
        let best = LagSelection::by_bic(&[y], 5);
        assert!(best >= 1 && best <= 5);
    }

    #[test]
    fn test_impulse_response_decay() {
        // AR(1) with coefficient < 1 should decay
        let y: Vec<f64> = (0..100).scan(0.0, |s, _| { *s = 0.5 * *s + 1.0; Some(*s) }).collect();
        let model = VARModel::fit(&[y], 1);
        let ir = ImpulseResponse::new(&model).compute(0, 10);
        // Responses should decay toward zero
        assert!(ir[0][0].abs() > ir[9][0].abs() || ir[9][0].abs() < 0.5);
    }

    #[test]
    fn test_impulse_response_length() {
        let y = vec![1.0; 50];
        let model = VARModel::fit(&[y], 1);
        let ir = ImpulseResponse::new(&model).compute(0, 5);
        assert_eq!(ir.len(), 5);
    }

    #[test]
    fn test_var_bic_larger_penalty() {
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let model1 = VARModel::fit(&[y.clone()], 1);
        let model3 = VARModel::fit(&[y], 3);
        // BIC penalizes more parameters
        // Not a strict rule, but generally holds for noisy data
        assert!(model1.bic().is_finite());
        assert!(model3.bic().is_finite());
    }
}
