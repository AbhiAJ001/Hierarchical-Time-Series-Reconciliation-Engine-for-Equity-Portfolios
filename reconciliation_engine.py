"""
Hierarchical Time-Series Reconciliation Engine for Equity Portfolios
A Comparative Study of Auto-ARIMA (with Fourier Terms), Holt-Winters ETS, and XGBoost
across Bottom-Up, Top-Down, Middle-Out, and Optimal MinT Reconciliation.
"""

# --- Step / Cell 4 ---
# Enable inline plotting for Jupyter notebook environment
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Import statistical and classical time-series analysis tools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import lag_plot
from scipy import stats
import pmdarima as pm

# Import machine learning and evaluation metrics
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warning logs for clean cell execution
warnings.filterwarnings("ignore")

# Set default plotting styles
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Libraries imported successfully.")

# --- Step / Cell 6 ---
# STEP 2: Download Stock Data from Yahoo Finance API
# What: Define NSE tickers for HDFC Bank, ICICI Bank, TCS, and Infosys.
# Why: These represent the base assets (bottom-level nodes) of our portfolio hierarchy.
# Need: We need daily adjusted closing prices to perform our hierarchical analysis.

tickers = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS"
}

# Fetch historical data dynamically across a multi-year range
raw_data = yf.download(
    tickers=list(tickers.values()),
    start="2022-01-01",
    end="2026-06-01",
    auto_adjust=True,
    progress=False
)["Close"]

# Clean up column headers to match our internal hierarchical asset labels
raw_df = raw_data.rename(columns={v: k for k, v in tickers.items()})
print(f"Raw dataset loaded with shape: {raw_df.shape}")

# --- Step / Cell 7 ---
raw_df.head()

# --- Step / Cell 8 ---
raw_df.info()

# --- Step / Cell 10 ---
# STEP 3: Temporal Sorting and Index Cleaning
# What: Strip timezones from the index and sort chronologically.
# Why: Financial time-series models require chronological consistency.
# Need: Ensuring that dates strictly increase without overlaps is crucial for lag generation.

raw_df.index = pd.to_datetime(raw_df.index).tz_localize(None)
raw_df = raw_df.sort_index(ascending=True)

print(f"Date Range: {raw_df.index.min().strftime('%Y-%m-%d')} to {raw_df.index.max().strftime('%Y-%m-%d')}")
print(f"Is Chronologically Monotonic Increasing? {raw_df.index.is_monotonic_increasing}")

# --- Step / Cell 11 ---
raw_df.head()

# --- Step / Cell 13 ---
# STEP 4: Enforce Continuous Business Day Frequency & Initial Portfolio Normalization
# What: Resample the index to regular Business Days ('B') and impute missing values.
# Why: Real-world market closures create gaps. We fill these gaps using forward/backward fill.
# Need: We normalize all initial stock prices to 100 to allow balanced portfolio tracking.

duplicate_dates = raw_df.index.duplicated().sum()
initial_missing = raw_df.isna().sum().to_dict()
print(f" - Duplicate timestamps found: {duplicate_dates}")
print(f" - Initial missing values per stock: {initial_missing}")

# Remove duplicates and fill gaps to establish a regular time series structure
raw_df = raw_df[~raw_df.index.duplicated(keep="first")]
clean_bottom_df = raw_df.asfreq("B").ffill().bfill()

# Apply base-value scale factor normalisation
bottom_cols = ["TCS", "INFY", "HDFC", "ICICI"]
clean_bottom_df = (clean_bottom_df[bottom_cols] / clean_bottom_df[bottom_cols].iloc[0]) * 100.0

print(f" - Enforced Time Frequency: '{clean_bottom_df.index.freqstr}' (Business Daily)")
print(f" - Post-cleaning missing values: {clean_bottom_df.isna().sum().sum()}")

# --- Step / Cell 16 ---
hierarchy_table = pd.DataFrame([
    {"Level": "Level 0 (Top)",    "Node": "Total", "Parent": "None",  "Components": "IT + Bank (TCS + INFY + HDFC + ICICI)"},
    {"Level": "Level 1 (Middle)", "Node": "IT",    "Parent": "Total", "Components": "TCS + INFY"},
    {"Level": "Level 1 (Middle)", "Node": "Bank",  "Parent": "Total", "Components": "HDFC + ICICI"},
    {"Level": "Level 2 (Bottom)", "Node": "TCS",   "Parent": "IT",    "Components": "Base Asset (TCS.NS)"},
    {"Level": "Level 2 (Bottom)", "Node": "INFY",  "Parent": "IT",    "Components": "Base Asset (INFY.NS)"},
    {"Level": "Level 2 (Bottom)", "Node": "HDFC",  "Parent": "Bank",  "Components": "Base Asset (HDFCBANK.NS)"},
    {"Level": "Level 2 (Bottom)", "Node": "ICICI", "Parent": "Bank",  "Components": "Base Asset (ICICIBANK.NS)"},
])
print(hierarchy_table.to_string(index=False))

# --- Step / Cell 18 ---
# STEP 6: Summing Matrix (S) Construction & Historical Coherence Verification
# What: Build the binary Summing Matrix S mapping bottom nodes to aggregate sectors and total portfolio.
# Why: Linear algebraic reconciliation models rely on the structural identity: Y_t = S * B_t.
# Need: Validate that historical levels add up perfectly to confirm algebraic consistency.

S = np.array([
    [1, 1, 1, 1],  # Row 1: Total Portfolio = TCS + INFY + HDFC + ICICI
    [1, 1, 0, 0],  # Row 2: IT Sector = TCS + INFY
    [0, 0, 1, 1],  # Row 3: Banking Sector = HDFC + ICICI
    [1, 0, 0, 0],  # Row 4: TCS
    [0, 1, 0, 0],  # Row 5: INFY
    [0, 0, 1, 0],  # Row 6: HDFC
    [0, 0, 0, 1]   # Row 7: ICICI
], dtype=float)

all_nodes = ["Total", "IT", "Bank", "TCS", "INFY", "HDFC", "ICICI"]

# Perform structural projection: Y = (S @ B^T)^T
full_values = (S @ clean_bottom_df[bottom_cols].values.T).T
y_df = pd.DataFrame(full_values, index=clean_bottom_df.index, columns=all_nodes)

# Check for mathematical deviations
hist_sector_diff = np.max(np.abs(y_df["Total"] - (y_df["IT"] + y_df["Bank"])))
hist_stock_diff = np.max(np.abs(y_df["Total"] - y_df[bottom_cols].sum(axis=1)))
print(f" - Max Historical Discrepancy (Total vs Sectors): {hist_sector_diff:.12f}")
print(f" - Max Historical Discrepancy (Total vs Stocks):  {hist_stock_diff:.12f}")
print("\nCleaned Hierarchical Dataset Preview (First 5 Rows):")
display(y_df.head().round(2))
display(y_df.describe().round(2))

# --- Step / Cell 20 ---
# STEP 7: Stationarity Tests (Augmented Dickey-Fuller & KPSS)
# What: Iterate through hierarchy nodes to compute summary stats, ADF, and KPSS statistics.
# Why: To determine if the time-series require differencing (I(1)) before fitting ARIMA.
# Need: Identifying unit-roots tells us whether we should fit models on raw levels or diffs.

eda_rows = []
for col in all_nodes:
    # ADF & KPSS on raw level series
    adf_level_stat, adf_level_p, *_ = adfuller(y_df[col], autolag="AIC")
    kpss_level_stat, kpss_level_p, *_ = kpss(y_df[col], regression="c", nlags="auto")

    # ADF & KPSS on first-differenced series (d = 1)
    diff_s = y_df[col].diff().dropna()
    adf_diff_stat, adf_diff_p, *_ = adfuller(diff_s, autolag="AIC")
    kpss_diff_stat, kpss_diff_p, *_ = kpss(diff_s, regression="c", nlags="auto")

    eda_rows.append({
        "Series": col,
        "Mean": round(y_df[col].mean(), 2),
        "Std Dev": round(y_df[col].std(), 2),
        "Min": round(y_df[col].min(), 2),
        "Max": round(y_df[col].max(), 2),
        "ADF p (Level)": round(adf_level_p, 4),
        "KPSS p (Level)": round(kpss_level_p, 4),
        "ADF p (1st Diff)": f"{adf_diff_p:.2e}",
        "KPSS p (1st Diff)": round(kpss_diff_p, 4),
        "Stationarity": "I(1) Non-Stationary" if (adf_level_p > 0.05 and adf_diff_p < 0.05) else "Stationary"
    })

eda_summary_df = pd.DataFrame(eda_rows)
print(eda_summary_df.to_string(index=False))

# --- Step / Cell 21 ---
# --- PLOT 1: Multi-Level Historical Time Series Plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(y_df.index, y_df["Total"], color="black", lw=2, label="Total Portfolio (Level 0)")
axes[0].set_title("Level 0: Total Portfolio Historical Trajectory")
axes[0].legend(loc="upper left")

axes[1].plot(y_df.index, y_df["IT"], label="IT Sector", color="tab:blue")
axes[1].plot(y_df["Bank"].index, y_df["Bank"], label="Banking Sector", color="tab:orange")
axes[1].set_title("Level 1: Sector Trajectories (IT vs Banking)")
axes[1].legend(loc="upper left")

for c in bottom_cols:
    axes[2].plot(y_df.index, y_df[c], label=c, alpha=0.85)
axes[2].set_title("Level 2: Individual NSE Equity Trajectories (Normalized to 100 at t=0)")
axes[2].legend(loc="upper left", ncol=4)
plt.tight_layout()
plt.savefig("step7_hierarchy_eda_plots.png", dpi=150)
plt.show()

# --- Step / Cell 23 ---
# PLOT 2: Seasonal Plot, Seasonal Subseries Plot & Lag Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 2a. Yearly Seasonal Plot
temp_df = y_df[["Total"]].copy()
temp_df["Year"] = temp_df.index.year
temp_df["Month"] = temp_df.index.month
monthly_piv = temp_df.groupby(["Month", "Year"])["Total"].mean().unstack()
monthly_piv.plot(ax=axes[0], marker="o", lw=1.5)
axes[0].set_title("Seasonal Plot: Monthly Total Portfolio by Year")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Portfolio Value")

# 2b. Day-of-Week Seasonal Subseries Plot
temp_df["DayName"] = temp_df.index.day_name()
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
day_groups = [temp_df[temp_df["DayName"] == d]["Total"].values for d in days_order]
axes[1].boxplot(day_groups, labels=["Mon", "Tue", "Wed", "Thu", "Fri"], patch_artist=True)
axes[1].set_title("Seasonal Subseries Plot (Trading Day of Week)")
axes[1].set_ylabel("Portfolio Value")

# 2c. Lag Plot (t vs t+1)
lag_plot(y_df["Total"], lag=1, ax=axes[2], c="tab:purple", alpha=0.5)
axes[2].set_title("Lag-1 Autocorrelation Plot: Total Portfolio")
plt.tight_layout()
plt.savefig("step7_seasonal_and_lag_plots.png", dpi=150)
plt.show()

# --- Step / Cell 25 ---
# PLOT 3: 4-Panel Dataset Overview (Reference Notebook Style)
total_s = y_df["Total"]
fig, ax = plt.subplots(2, 2, figsize=(15, 9))

# (1) Raw series + fitted straight-line trend
t_idx = np.arange(len(total_s))
slope, intercept = np.polyfit(t_idx, total_s.values, 1)
ax[0, 0].plot(total_s, label="Total Portfolio", color="tab:blue")
ax[0, 0].plot(total_s.index, intercept + slope * t_idx, "r--", label=f"Linear trend ({slope*252:.2f} units/year)")
ax[0, 0].set_title("(1) Raw Total Portfolio with Linear Trend")
ax[0, 0].set_xlabel("Date"); ax[0, 0].set_ylabel("Capital Units"); ax[0, 0].legend()

# (2) Histogram
ax[0, 1].hist(total_s, bins=30, color="tab:blue", alpha=0.7, edgecolor="white")
ax[0, 1].axvline(total_s.mean(), color="red", ls="--", label=f"Mean = {total_s.mean():.1f}")
ax[0, 1].set_title("(2) Distribution of Total Portfolio Values")
ax[0, 1].set_xlabel("Capital Units"); ax[0, 1].set_ylabel("Frequency"); ax[0, 1].legend()

# (3) Rolling mean and std (21 trading days = 1 month)
ax[1, 0].plot(total_s.rolling(21).mean(), color="red", label="21-day rolling mean")
ax[1, 0].set_ylabel("Rolling mean", color="red")
ax2 = ax[1, 0].twinx()
ax2.plot(total_s.rolling(21).std(), color="black", alpha=0.7, label="21-day rolling std"); ax2.grid(False)
ax2.set_ylabel("Rolling std")
ax[1, 0].set_title("(3) 21-Day Rolling Mean (red) and Rolling Std (black)")
ax[1, 0].set_xlabel("Date")

# (4) Yearly Average & YoY Change
yearly_port = total_s.resample("YS").mean()
ax[1, 1].bar(yearly_port.index.year.astype(str), yearly_port.values, color="teal", alpha=0.85)
ax[1, 1].set_title("(4) Yearly Average Portfolio Value")
ax[1, 1].set_xlabel("Year"); ax[1, 1].set_ylabel("Capital Units")

plt.suptitle("Dataset Overview – Total Equity Portfolio (2022–2026)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# --- Step / Cell 28 ---
decomp = seasonal_decompose(y_df["Total"], model="additive", period=21)
fig = decomp.plot()
fig.set_size_inches(12, 8)
fig.suptitle("Additive Decomposition of Total Portfolio (Period = 21 Trading Days)", y=1.01)
plt.tight_layout()
plt.show()

var_resid = decomp.resid.dropna().var()
strength_trend = max(0, 1 - var_resid / (decomp.trend + decomp.resid).dropna().var())
strength_season = max(0, 1 - var_resid / (decomp.seasonal + decomp.resid).dropna().var())
print(f"Strength of Trend       : {strength_trend:.4f}   (0 = none, 1 = very strong)")
print(f"Strength of Seasonality : {strength_season:.4f}   (0 = none, 1 = very strong)")

# --- Step / Cell 31 ---
stages = {
    "Original Level Series (d = 0)": y_df["Total"],
    "First-Differenced Series (d = 1)": y_df["Total"].diff().dropna(),
}

fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for i, (label, s) in enumerate(stages.items()):
    p_adf = adfuller(s, autolag="AIC")[1]
    status = "STATIONARY" if p_adf < 0.05 else "NON-STATIONARY"
    col_title = "green" if p_adf < 0.05 else "red"

    ax[i, 0].plot(s, lw=0.9, color="tab:blue")
    ax[i, 0].axhline(s.mean(), color="black", ls="--", lw=0.8)
    ax[i, 0].set_title(f"{label}\nADF p = {p_adf:.4e} -> {status}", color=col_title)

    ax[i, 1].plot(s.rolling(21).mean(), color="red", label="21-day Rolling Mean")
    ax[i, 1].plot(s.rolling(21).std(), color="black", label="21-day Rolling Std")
    ax[i, 1].set_title(f"{label}: Rolling Mean & Std")
    ax[i, 1].legend(fontsize=8)

    plot_acf(s, lags=40, ax=ax[i, 2])
    ax[i, 2].set_title(f"{label}: ACF")
    for L in [5, 10, 15, 21]:
        ax[i, 2].axvline(L, color="red", ls=":", alpha=0.5)

plt.suptitle("Making the Total Portfolio Series Stationary – Before vs. After Differencing", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# PACF of Original vs Stationary Differenced Series
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
plot_pacf(y_df["Total"], lags=40, ax=ax[0], method="ywm")
ax[0].set_title("PACF – Original Non-Stationary Series (Huge Spike at Lag 1 ≈ 1.0)")
plot_pacf(y_df["Total"].diff().dropna(), lags=40, ax=ax[1], method="ywm")
ax[1].set_title("PACF – Stationary First-Differenced Series (d = 1)")
plt.tight_layout()
plt.show()

# --- Step / Cell 35 ---
h = 20 # 20-trading-day
train_df = y_df.iloc[:-h].copy()
test_df = y_df.iloc[-h:].copy()
print(f"[STEP 8] Chronological Train/Test Split (No Shuffling):")
print(f" - Training Set: {len(train_df)} obs ({train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')})")
print(f" - Testing Set:  {len(test_df)} obs ({test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})")

# Visual Train-Test Split Plot (Zoomed to Recent Period for Clarity)
plt.figure(figsize=(12, 4.5))
plt.plot(train_df["Total"].iloc[-120:], label="Train (Last 120 Trading Days)", color="tab:blue", lw=1.8)
plt.plot(test_df["Total"], label="Unseen Test Horizon (h = 20 Trading Days)", color="tab:orange", lw=2.5, marker="o", ms=4)
plt.title("Chronological Train–Test Split (Level 0: Total Portfolio)")
plt.xlabel("Date"); plt.ylabel("Portfolio Value (Capital Units)"); plt.legend()
plt.tight_layout()
plt.show()

# --- Step / Cell 37 ---
print("[STEP 9 & 10] Training Base Forecasting Models Across All 7 Hierarchy Nodes...")

def make_fourier_features(n_obs, periods=(5, 21), orders=(2, 2)):
    t = np.arange(1, n_obs + 1)
    feats = {}
    for m, k_max in zip(periods, orders):
        for k in range(1, k_max + 1):
            feats[f"sin_m{m}_k{k}"] = np.sin(2 * np.pi * k * t / m)
            feats[f"cos_m{m}_k{k}"] = np.cos(2 * np.pi * k * t / m)
    return pd.DataFrame(feats)

fourier_all = make_fourier_features(len(y_df))
X_fourier_train = fourier_all.iloc[:-h].values
X_fourier_test = fourier_all.iloc[-h:].values

def train_and_forecast_xgboost(series, horizon, lags=5):
    """Builds tabular lag + rolling volatility features and forecasts recursively."""
    vals = series.values
    X_rows, y_rows = [], []
    for i in range(lags, len(vals)):
        window = vals[i - lags:i]
        features = list(window[::-1]) + [np.mean(window), np.std(window)]
        X_rows.append(features)
        y_rows.append(vals[i])
    X_arr, y_arr = np.array(X_rows), np.array(y_rows)

    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0
    )
    model.fit(X_arr, y_arr)
    in_sample_resid = y_arr - model.predict(X_arr)

    # Recursive multi-step forecast
    history = list(vals)
    preds = []
    for _ in range(horizon):
        win = np.array(history[-lags:])
        feat = np.array(list(win[::-1]) + [np.mean(win), np.std(win)]).reshape(1, -1)
        nxt = float(model.predict(feat)[0])
        preds.append(nxt)
        history.append(nxt)
    return np.array(preds), in_sample_resid

base_forecasts = {"Auto_ARIMA": [], "Holt_Winters_ETS": [], "XGBoost": []}
base_residuals = {"Auto_ARIMA": [], "Holt_Winters_ETS": [], "XGBoost": []}
model_order_info = []

for col in all_nodes:
    s_train = train_df[col]

    # --- Model 1: Auto ARIMA with Fourier Exogenous Terms ---
    arima_mod = pm.auto_arima(
        s_train,
        X=X_fourier_train,
        start_p=0, max_p=2,
        start_q=0, max_q=2,
        d=None,
        seasonal=False,
        information_criterion="aicc",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    fc_arima = arima_mod.predict(n_periods=h, X=X_fourier_test).values
    res_arima = arima_mod.resid()[5:]  # Align length with 5-lag XGBoost

    # --- Model 2: Holt-Winters Exponential Smoothing (Syllabus Unit 2) ---
    ets_mod = ExponentialSmoothing(
        s_train,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=5,
        initialization_method="estimated"
    ).fit(optimized=True)
    fc_ets = ets_mod.forecast(h).values
    res_ets = ets_mod.resid.values[5:]

    # --- Model 3: Out-of-Syllabus XGBoost Regressor ---
    fc_xgb, res_xgb = train_and_forecast_xgboost(s_train, horizon=h, lags=5)

    base_forecasts["Auto_ARIMA"].append(fc_arima)
    base_forecasts["Holt_Winters_ETS"].append(fc_ets)
    base_forecasts["XGBoost"].append(fc_xgb)

    base_residuals["Auto_ARIMA"].append(res_arima)
    base_residuals["Holt_Winters_ETS"].append(res_ets)
    base_residuals["XGBoost"].append(res_xgb)

    model_order_info.append({
        "Node": col,
        "Auto_ARIMA Order": f"ARIMA{arima_mod.order} + Fourier",
        "Auto_ARIMA AIC": round(arima_mod.aic(), 2),
        "Holt-Winters ETS AIC": round(ets_mod.aic, 2),
        "XGBoost Spec": "XGB(lags=5, depth=3, n=150)"
    })
    print(f" - Node {col:<6} | Auto-ARIMA{arima_mod.order} (AIC={arima_mod.aic():.1f}) | ETS(Add,Damped,m=5) (AIC={ets_mod.aic:.1f}) | XGBoost trained.")

for k in base_forecasts:
    base_forecasts[k] = np.array(base_forecasts[k])          # Shape: (7, h)
    base_residuals[k] = np.column_stack(base_residuals[k])  # Shape: (T-5, 7)

print("\nSummary of Fitted Base Models by Hierarchy Node:")
display(pd.DataFrame(model_order_info))

# --- Step / Cell 40 ---
def residual_analysis(resid_array, index_dates, name):
    """4-panel residual diagnostic plot + Ljung-Box, Shapiro-Wilk & Jarque-Bera tests."""
    resid = pd.Series(resid_array, index=index_dates).dropna()
    fig, ax = plt.subplots(2, 2, figsize=(14, 7.5))

    ax[0, 0].plot(resid, lw=0.8, color="tab:blue")
    ax[0, 0].axhline(0, color="red", ls="--")
    ax[0, 0].set_title("Residuals over Time (Level 0: Total)")
    ax[0, 0].set_ylabel("Capital Units")

    ax[0, 1].hist(resid, bins=30, density=True, alpha=0.7, edgecolor="white", color="tab:blue")
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax[0, 1].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), "r", lw=2, label="Normal curve")
    ax[0, 1].set_title("Histogram of Residuals")
    ax[0, 1].legend()

    plot_acf(resid, lags=36, ax=ax[1, 0])
    ax[1, 0].set_title("ACF of Residuals")

    sm.qqplot(resid, line="s", ax=ax[1, 1])
    ax[1, 1].set_title("QQ Plot of Residuals")

    plt.suptitle(f"Residual Analysis – {name} (Level 0: Total Portfolio)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    sw_p = stats.shapiro(resid).pvalue
    jb_p = stats.jarque_bera(resid).pvalue
    print(f"Residual Diagnostics for {name}:")
    print(f"  Mean of residuals    : {resid.mean():.4f}  (should be close to 0)")
    print(f"  Std of residuals     : {resid.std():.4f}")
    print(f"  Ljung-Box p (lag 10) : {lb_p:.4f} -> {'White noise (PASS)' if lb_p > 0.05 else 'Some autocorrelation remains (REVIEW)'}")
    print(f"  Shapiro-Wilk p       : {sw_p:.4e} | Jarque-Bera p : {jb_p:.4e}\n")
    return {
        "Resid Mean": round(resid.mean(), 4),
        "Resid Std": round(resid.std(), 4),
        "Ljung-Box p (Lag 10)": round(lb_p, 4),
        "White Noise? (LB p>0.05)": "Yes" if lb_p > 0.05 else "No",
        "Shapiro-Wilk p": f"{sw_p:.2e}",
        "Jarque-Bera p": f"{jb_p:.2e}"
    }

resid_dates = train_df.index[5:]
resid_summary = {}
for m_name in ["Auto_ARIMA", "Holt_Winters_ETS", "XGBoost"]:
    # Index 0 corresponds to Level 0 ('Total')
    resid_summary[m_name] = residual_analysis(base_residuals[m_name][:, 0], resid_dates, m_name)

print("Side-by-Side Residual Diagnostic Table (Level 0: Total Portfolio):")
display(pd.DataFrame(resid_summary).T)

# --- Step / Cell 42 ---
print("[STEP 11] Pre-Reconciliation Base Forecast Coherence Check:")
coherence_before_rows = []
for model_name, y_hat in base_forecasts.items():
    # Discrepancy 1: Total - (IT + Bank)
    gap_l0_l1 = np.max(np.abs(y_hat[0] - (y_hat[1] + y_hat[2])))
    # Discrepancy 2: Total - (TCS + INFY + HDFC + ICICI)
    gap_l0_l2 = np.max(np.abs(y_hat[0] - np.sum(y_hat[3:], axis=0)))
    # Discrepancy 3: IT - (TCS + INFY)
    gap_it = np.max(np.abs(y_hat[1] - (y_hat[3] + y_hat[4])))
    # Discrepancy 4: Bank - (HDFC + ICICI)
    gap_bank = np.max(np.abs(y_hat[2] - (y_hat[5] + y_hat[6])))

    coherence_before_rows.append({
        "Model": model_name,
        "Max |Total - (IT+Bank)|": round(gap_l0_l1, 4),
        "Max |Total - Sum(Stocks)|": round(gap_l0_l2, 4),
        "Max |IT - (TCS+INFY)|": round(gap_it, 4),
        "Max |Bank - (HDFC+ICICI)|": round(gap_bank, 4),
        "Is Coherent?": "NO (Incoherent)" if gap_l0_l2 > 1e-6 else "YES"
    })
display(pd.DataFrame(coherence_before_rows))

# --- Step / Cell 43 ---
# --- PLOT: Visual Proof of Base Forecast Incoherence Across Horizon ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
for idx, (model_name, y_hat) in enumerate(base_forecasts.items()):
    ax = axes[idx]
    ax.plot(test_df.index, y_hat[0], "k-o", lw=2, label="Direct Total Base Forecast")
    ax.plot(test_df.index, y_hat[1] + y_hat[2], "b--s", lw=1.8, label="Sum of Sectors (IT + Bank)")
    ax.plot(test_df.index, np.sum(y_hat[3:], axis=0), "r:^", lw=1.8, label="Sum of 4 Bottom Stocks")
    ax.set_title(f"{model_name}: Pre-Reconciliation Gap")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("step11_pre_reconciliation_incoherence.png", dpi=150)
plt.show()

# --- Step / Cell 46 ---
print("[STEPS 12-16] Computing Projection Matrices (BU, TD, MO, MinT-OLS, MinT-WLS)...")

# Step 12: Bottom-Up Projection Matrix P_BU (4 x 7)
P_BU = np.zeros((4, 7))
P_BU[:, 3:] = np.eye(4)

# Step 13: Top-Down Historical Proportions Matrix P_TD (4 x 7)
hist_top_props = [np.mean(train_df[c] / train_df["Total"]) for c in bottom_cols]
P_TD = np.zeros((4, 7))
P_TD[:, 0] = hist_top_props  # Disaggregates Level 0 ('Total') directly to bottom stocks

# Step 14: Middle-Out Historical Intra-Sector Proportions Matrix P_MO (4 x 7)
p_tcs_it = np.mean(train_df["TCS"] / train_df["IT"])
p_infy_it = np.mean(train_df["INFY"] / train_df["IT"])
p_hdfc_bank = np.mean(train_df["HDFC"] / train_df["Bank"])
p_icici_bank = np.mean(train_df["ICICI"] / train_df["Bank"])

P_MO = np.zeros((4, 7))
P_MO[0, 1] = p_tcs_it     # TCS from IT
P_MO[1, 1] = p_infy_it    # INFY from IT
P_MO[2, 2] = p_hdfc_bank  # HDFC from Bank
P_MO[3, 2] = p_icici_bank # ICICI from Bank

# Step 15a: Optimal MinT-OLS Projection Matrix P_OLS (4 x 7)
P_OLS = np.linalg.pinv(S.T @ S) @ S.T

# Dictionary to store all base and reconciled forecasts: reconciled_forecasts[model][method] = (7, h) array
reconciled_forecasts = {}

for model_name, y_hat in base_forecasts.items():
    resid_mat = base_residuals[model_name]
    # Step 15b: Optimal MinT-WLS (Variance Scaling using in-sample error variances)
    diag_vars = np.var(resid_mat, axis=0)
    W_inv = np.diag(1.0 / np.maximum(diag_vars, 1e-6))
    P_WLS = np.linalg.pinv(S.T @ W_inv @ S) @ (S.T @ W_inv)

    reconciled_forecasts[model_name] = {
        "Base (Incoherent)": y_hat,
        "Bottom-Up (BU)":    S @ P_BU @ y_hat,
        "Top-Down (TD)":     S @ P_TD @ y_hat,
        "Middle-Out (MO)":   S @ P_MO @ y_hat,
        "MinT-OLS":          S @ P_OLS @ y_hat,
        "MinT-WLS":          S @ P_WLS @ y_hat,
    }

print("All reconciled forecast matrices constructed across all 7 hierarchy levels.")

# --- Step / Cell 48 ---
print("[STEP 17] Post-Reconciliation Coherence Verification Check:")
coherence_after_rows = []
for model_name, methods_dict in reconciled_forecasts.items():
    for method_name, y_tilde in methods_dict.items():
        if method_name == "Base (Incoherent)":
            continue
        # Verify y_tilde == S @ b_tilde across all levels
        max_coherence_error = np.max(np.abs(y_tilde - (S @ y_tilde[3:])))
        coherence_after_rows.append({
            "Model": model_name,
            "Reconciliation Method": method_name,
            "Max Hierarchy Coherence Error": f"{max_coherence_error:.2e}",
            "Coherence Verified?": "YES (Strictly Coherent)" if max_coherence_error < 1e-8 else "NO"
        })
display(pd.DataFrame(coherence_after_rows))

# --- Step / Cell 50 ---
print("[STEP 18 & 19] Out-of-Sample Evaluation (MAE, RMSE, MAPE) Across Hierarchy Levels:")
actual_mat = test_df.values.T  # Shape: (7, h)

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return mae, rmse, mape

evaluation_records = []

for model_name, methods_dict in reconciled_forecasts.items():
    for method_name, y_pred in methods_dict.items():
        # Overall average across all 7 series
        mae_all, rmse_all, mape_all = compute_metrics(actual_mat, y_pred)
        # Level-specific RMSEs
        _, rmse_l0, _ = compute_metrics(actual_mat[0:1], y_pred[0:1])   # Level 0: Total
        _, rmse_l1, _ = compute_metrics(actual_mat[1:3], y_pred[1:3])   # Level 1: Sectors
        _, rmse_l2, _ = compute_metrics(actual_mat[3:7], y_pred[3:7])   # Level 2: Stocks

        evaluation_records.append({
            "Model": model_name,
            "Method": method_name,
            "Overall MAE": round(mae_all, 3),
            "Overall RMSE": round(rmse_all, 3),
            "Overall MAPE (%)": round(mape_all, 3),
            "Level 0 (Total) RMSE": round(rmse_l0, 3),
            "Level 1 (Sector) RMSE": round(rmse_l1, 3),
            "Level 2 (Stock) RMSE": round(rmse_l2, 3),
        })

eval_df = pd.DataFrame(evaluation_records)
eval_df["Rank (by Overall RMSE)"] = eval_df["Overall RMSE"].rank(method="min").astype(int)
display(eval_df.sort_values("Overall RMSE").reset_index(drop=True))

best_row = eval_df.loc[eval_df["Overall RMSE"].idxmin()]
best_combo_label = f"{best_row['Model']} + {best_row['Method']}"
print(f"\nBest Performing Combination (Lowest Overall RMSE): {best_combo_label} (RMSE = {best_row['Overall RMSE']})")

# --- Step / Cell 51 ---
# --- PLOT: Grouped Bar Chart Comparing Reconciliation Accuracy ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

piv_rmse = eval_df.pivot(index="Method", columns="Model", values="Overall RMSE")
piv_rmse.loc[["Base (Incoherent)", "Bottom-Up (BU)", "Middle-Out (MO)", "MinT-OLS", "MinT-WLS"]].plot(
    kind="bar", ax=axes[0], rot=25
)
axes[0].set_title("Overall RMSE by Reconciliation Method (Excluding TD Outlier)")
axes[0].set_ylabel("RMSE")

piv_l0 = eval_df.pivot(index="Method", columns="Model", values="Level 0 (Total) RMSE")
piv_l0.loc[["Base (Incoherent)", "Bottom-Up (BU)", "Middle-Out (MO)", "MinT-OLS", "MinT-WLS"]].plot(
    kind="bar", ax=axes[1], rot=25
)
axes[1].set_title("Level 0 (Total Portfolio) RMSE by Reconciliation Method")
axes[1].set_ylabel("RMSE")

plt.tight_layout()
plt.savefig("step19_reconciliation_metrics_comparison.png", dpi=150)
plt.show()

# --- Step / Cell 53 ---
# PLOT 20.1: Actual vs. Base & Reconciled Forecasts for All 3 Models (Level 0: Total Portfolio)
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
test_dates = test_df.index

for idx, model_name in enumerate(["Auto_ARIMA", "Holt_Winters_ETS", "XGBoost"]):
    ax = axes[idx]
    methods_dict = reconciled_forecasts[model_name]
    ax.plot(test_dates, actual_mat[0], "k-o", lw=2.5, label="Actual Total")
    ax.plot(test_dates, methods_dict["Base (Incoherent)"][0], "--", lw=1.8, label="Base (Incoherent)")
    ax.plot(test_dates, methods_dict["Bottom-Up (BU)"][0], ":", lw=1.8, label="Bottom-Up")
    ax.plot(test_dates, methods_dict["Middle-Out (MO)"][0], "-.", lw=1.8, label="Middle-Out")
    ax.plot(test_dates, methods_dict["MinT-WLS"][0], "-", color="crimson", lw=2.2, label="MinT-WLS (Optimal)")
    ax.set_title(f"{model_name}: Level 0 (Total Portfolio)")
    ax.set_xlabel("Forecast Horizon Date")
    if idx == 0:
        ax.set_ylabel("Portfolio Value (Capital Units)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("step20_actual_vs_reconciled_forecasts.png", dpi=150)
plt.show()

# --- Step / Cell 54 ---
# PLOT 20.2: Side-by-Side Metric Bar Charts (MAE, RMSE, MAPE) for MinT-WLS Reconciled Models (Reference Style)
mint_wls_df = eval_df[eval_df["Method"] == "MinT-WLS"].set_index("Model")
best_m = mint_wls_df["Overall RMSE"].idxmin()

metric_cols = ["Overall MAE", "Overall RMSE", "Overall MAPE (%)"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, mcol in zip(ax, metric_cols):
    vals = mint_wls_df[mcol]
    colors = ["tab:green" if m == best_m else "tab:gray" for m in vals.index]
    bars = a.bar(vals.index, vals.values, color=colors, width=0.55)
    for b, v in zip(bars, vals.values):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    a.set_title(f"{mcol} (MinT-WLS Reconciled)")
    a.set_ylabel(mcol)

plt.suptitle("Model Comparison – All Metrics on Test Set (Green = Best Model)", fontsize=14, y=1.03)
plt.tight_layout()
plt.show()

# --- Step / Cell 55 ---
# PLOT 20.3: Absolute Forecast Error per Day on Test Set (Reference Step 13.3 Style)
plt.figure(figsize=(13, 4.5))
styles = {"Auto_ARIMA": "-", "Holt_Winters_ETS": "--", "XGBoost": "-."}
for m_name in ["Auto_ARIMA", "Holt_Winters_ETS", "XGBoost"]:
    pred_l0 = reconciled_forecasts[m_name]["MinT-WLS"][0]
    err_series = pd.Series(np.abs(actual_mat[0] - pred_l0), index=test_dates)
    mae_l0 = np.mean(err_series)
    plt.plot(err_series, ls=styles[m_name], marker="o", ms=4, lw=1.8, label=f"{m_name} + MinT-WLS (Level 0 MAE = {mae_l0:.2f})")

plt.title("Absolute Forecast Error per Trading Day on Test Set (Level 0: Total Portfolio)")
plt.xlabel("Test Horizon Date"); plt.ylabel("|Actual − Forecast| (Capital Units)"); plt.legend()
plt.tight_layout()
plt.show()

# PLOT 20.4: Actual vs. Predicted Scatter Plot Across All 7 Hierarchy Nodes (Reference Step 13.4 Style)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
lo, hi = actual_mat.min() - 5, actual_mat.max() + 5
for a, m_name in zip(ax, ["Auto_ARIMA", "Holt_Winters_ETS", "XGBoost"]):
    pred_all = reconciled_forecasts[m_name]["MinT-WLS"].flatten()
    act_all = actual_mat.flatten()
    a.scatter(act_all, pred_all, s=18, alpha=0.75, color="tab:blue")
    a.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect forecast (45° line)")
    r2 = 1 - np.sum((act_all - pred_all) ** 2) / np.sum((act_all - np.mean(act_all)) ** 2)
    a.set_title(f"{m_name} + MinT-WLS (R² = {r2:.4f})")
    a.set_xlabel("Actual Value (Capital Units)")
ax[0].set_ylabel("Predicted Value (Capital Units)")
ax[0].legend()
plt.suptitle("Actual vs. Reconciled Predicted Values Across All 7 Hierarchy Series", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# --- Step / Cell 57 ---
steps_fut = 10
future_idx = pd.date_range(y_df.index[-1] + pd.offsets.BDay(1), periods=steps_fut, freq="B")
fourier_fut_all = make_fourier_features(len(y_df) + steps_fut)
X_f_hist = fourier_fut_all.iloc[:len(y_df)].values
X_f_fut  = fourier_fut_all.iloc[len(y_df):].values

future_base_arima = []
future_resid_arima = []

for col in all_nodes:
    full_s = y_df[col]
    m_full = pm.auto_arima(
        full_s, X=X_f_hist, start_p=0, max_p=2, start_q=0, max_q=2,
        d=None, seasonal=False, information_criterion="aicc",
        stepwise=True, suppress_warnings=True, error_action="ignore"
    )
    future_base_arima.append(m_full.predict(n_periods=steps_fut, X=X_f_fut).values)
    future_resid_arima.append(m_full.resid()[1:])

future_base_arima = np.array(future_base_arima)
future_resid_mat = np.column_stack(future_resid_arima)

# Reconcile Future Forecasts via MinT-WLS
W_fut_inv = np.diag(1.0 / np.maximum(np.var(future_resid_mat, axis=0), 1e-6))
P_WLS_fut = np.linalg.pinv(S.T @ W_fut_inv @ S) @ (S.T @ W_fut_inv)
future_reconciled_mint = S @ P_WLS_fut @ future_base_arima

future_df = pd.DataFrame(future_reconciled_mint.T, index=future_idx, columns=all_nodes)
future_df.index.name = "Future Trading Date"
print("Reconciled Future 10-Trading-Day Portfolio Forecast (MinT-WLS):")
display(future_df.round(2))

plt.figure(figsize=(12, 4.5))
plt.plot(y_df["Total"].iloc[-60:], color="black", lw=2, label="Historical Total Portfolio (Last 60 Days)")
plt.plot(future_df.index, future_df["Total"], color="crimson", marker="o", ms=4, lw=2.2, label="MinT-WLS Future Forecast (Next 10 Days)")
plt.axvline(y_df.index[-1], color="gray", ls="--")
plt.title("Future 10-Trading-Day Reconciled Portfolio Forecast")
plt.xlabel("Date"); plt.ylabel("Portfolio Value (Capital Units)"); plt.legend()
plt.tight_layout()
plt.show()

# --- Step / Cell 59 ---
print("======================================================================")
print(" STEP 21: AUTOMATED INTERPRETATION & KEY OBSERVATIONS")
print("======================================================================")
print(f"1. Best Performing Framework: {best_row['Model']} with {best_row['Method']} "
      f"(Overall RMSE = {best_row['Overall RMSE']}, Overall MAPE = {best_row['Overall MAPE (%)']}%).")
print("2. Coherence Restoration: All independent base forecasts violated structural")
print("   additivity (Total != IT + Bank). Every reconciliation method reduced")
print("   structural incoherence to ~0.00e+00 (floating-point precision).")
print("3. MinT-WLS Advantage: Weighting by inverse residual variance down-weights")
print("   volatile individual equities and leverages the high signal-to-noise ratio")
print("   of sector and total portfolio series.")
