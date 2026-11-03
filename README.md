# Hierarchical Time-Series Reconciliation Engine for Equity Portfolios

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance%20%7C%20Time%20Series-purple.svg)]()

> **A Comparative Study of Auto-ARIMA (with Fourier Terms), Holt-Winters ETS, and XGBoost across Bottom-Up, Top-Down, Middle-Out, and Optimal MinT Reconciliation**

---

## 📌 Academic Context

* **Course:** Applied Time Series Analysis (B.Tech AIML, Semester VII)
* **Institution:** SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering (MPSTME), Shirpur Campus
* **Authors:**
  * Jayesh Patil (Roll: C233)
  * Abhijit Patil (Roll: C263)
  * Ritesh Pawar (Roll: C276)

---

## 📖 Executive Summary & Problem Formulation

In quantitative portfolio management, risk and asset managers require simultaneous forecasts at multiple aggregation levels: individual stocks, industry sectors, and total portfolio value. However, forecasting each node independently using univariate or machine learning models invariably produces **incoherent forecasts**—i.e., the sum of asset forecasts does not equal the sector forecast, and the sum of sector forecasts fails to match the total portfolio forecast:

$$\hat{Y}_{\text{Total}, t+h} \neq \sum \hat{Y}_{\text{Sector}, t+h} \neq \sum \hat{Y}_{\text{Stock}, t+h}$$

This engine implements and evaluates linear algebraic **Forecast Reconciliation** techniques that project independent base forecasts into a mathematically coherent subspace ($\tilde{\mathbf{y}}_{t+h} = \mathbf{S}\mathbf{P}\hat{\mathbf{y}}_{t+h} = \mathbf{S}\tilde{\mathbf{b}}_{t+h}$), guaranteeing exact cross-sectional additivity across all levels of the hierarchy.

---

## 🏗️ Portfolio Hierarchy Architecture

The project models a 3-level hierarchical equity portfolio constructed using official National Stock Exchange (NSE) market data:

```
                          Level 0: Total Portfolio
                                     |
               +---------------------+---------------------+
               |                                           |
      Level 1: IT Sector                         Level 1: Banking Sector
               |                                           |
       +-------+-------+                           +-------+-------+
       |               |                           |               |
Level 2: TCS    Level 2: INFY              Level 2: HDFCBANK   Level 2: ICICIBANK
```

### Summing Matrix $\mathbf{S}$ ($7 \times 4$)

Let $\mathbf{b}_t = [y_{\text{TCS}, t}, y_{\text{INFY}, t}, y_{\text{HDFCBANK}, t}, y_{\text{ICICIBANK}, t}]^T$ represent the $m=4$ bottom-level series. The full $n=7$ node vector $\mathbf{y}_t$ is given by:

$$\mathbf{y}_t = \mathbf{S} \mathbf{b}_t$$

$$\mathbf{S} = \begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

---

## 🔬 Base Forecasting Paradigms

Across all 7 nodes of the hierarchy, 3 distinct forecasting paradigms are trained and evaluated:

1. **Auto-ARIMA + Fourier Terms** *(Core Syllabus)*:
   - Captures $I(1)$ differenced trends and short-term ARMA dynamics.
   - External Fourier harmonic regressors model multi-cycle calendar seasonality (5-day weekly trading cycles and 21-day monthly trading cycles).
2. **Holt-Winters ETS** *(In-Syllabus Benchmark)*:
   - Additive error, damped linear trend, and additive seasonal state-space formulation.
3. **XGBoost Regressor** *(Out-of-Syllabus ML Benchmark)*:
   - Engineered autoregressive lag features ($t-1, \dots, t-5$), rolling mean windows (5, 21 days), and rolling standard deviations to capture volatility clustering.

---

## 📐 Reconciliation Methodologies

| Method | Projection Matrix $\mathbf{P}$ | Key Characteristics |
|---|---|---|
| **Bottom-Up (BU)** | $\mathbf{P} = [\mathbf{0}_{4 \times 3} \mid \mathbf{I}_{4 \times 4}]$ | Aggregates bottom-level base forecasts upwards; ignores upper-level forecasts. |
| **Top-Down (TD)** | Proportion vector $\mathbf{p}$ based on historical averages | Disaggregates Level 0 forecasts downward; prone to distorting asset dynamics. |
| **Middle-Out (MO)** | Level 1 sector forecasts aggregated up & disaggregated down | Balances sectoral macro signals with individual equity trends. |
| **MinT-OLS** | $\mathbf{P} = (\mathbf{S}^T \mathbf{S})^{-1} \mathbf{S}^T$ | Ordinary least squares projection assuming spherical errors ($\mathbf{W} = \mathbf{I}$). |
| **MinT-WLS** | $\mathbf{P} = (\mathbf{S}^T \hat{\mathbf{W}}^{-1} \mathbf{S})^{-1} \mathbf{S}^T \hat{\mathbf{W}}^{-1}$ | Weighted least squares using diagonal variance weights of in-sample residuals. |

---

## 📊 End-to-End Workflow Pipeline

1. **Environment Setup & Dependencies:** Configuration of computational environment.
2. **Data Acquisition:** Automated retrieval of adjusted close prices via `yfinance` from the NSE (`2022-01-01` to `2026-06-01`).
3. **Data Preprocessing:** Datetime indexing, chronological sorting, forward-filling corporate actions, and business-day (`'B'`) frequency alignment.
4. **Hierarchy & Summing Matrix:** Formalization of summing matrix $\mathbf{S}$ and empirical verification of historical additivity.
5. **Exploratory Data Analysis:** Classical additive decomposition, ADF and KPSS unit root tests, differencing, and ACF/PACF plots.
6. **Chronological Splitting:** 20-trading-day out-of-sample test horizon.
7. **Model Training & Forecasting:** Independent base forecast generation across all 7 nodes.
8. **Residual Diagnostics:** Ljung-Box test for autocorrelation, Shapiro-Wilk test for normality, and standardized residual distribution checks.
9. **Pre-Reconciliation Coherence Audit:** Measurement of coherence gap ($\|\hat{\mathbf{y}} - \mathbf{S}\hat{\mathbf{b}}\|_1$).
10. **Projection Matrix Computation:** Computation of BU, TD, MO, MinT-OLS, and MinT-WLS projection matrices.
11. **Post-Reconciliation Coherence Audit:** Verification that residual coherence errors are zero machine epsilon ($\approx 0.00$).
12. **Out-of-Sample Performance Evaluation:** Calculation of MAE, RMSE, and MAPE across individual nodes and hierarchy tiers.
13. **Visualization & Diagnostic Dashboard:** Forecast trajectory plots, error degradation curves, scatter plots, and future projections.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Recommended: Virtual environment (`venv` or `conda`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbhiAJ001/Hierarchical-Time-Series-Reconciliation-Engine-for-Equity-Portfolios.git
   cd Hierarchical-Time-Series-Reconciliation-Engine-for-Equity-Portfolios
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

* **Interactive Notebook:**
  Launch JupyterLab or Jupyter Notebook to view interactive plots and step-by-step diagnostic outputs:
  ```bash
  jupyter notebook ATSA_Project_Hierarchical_Time_Series_Reconciliation_Engine_for_Equity_Portfolios.ipynb
  ```

* **Standalone Python Engine:**
  Execute the complete workflow via command line:
  ```bash
  python reconciliation_engine.py
  ```

---

## 🔒 Security & Quality Assurance

- **Zero Hardcoded Credentials:** Audited against API key leaks, private keys, authentication tokens, and user credentials.
- **Reproducible Pipeline:** Fixed random seeds across stochastic optimizers and machine learning models.
- **Clean Artifact Policy:** Production code, documentation, and dependencies committed without ephemeral build artifacts or HTML exports.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
