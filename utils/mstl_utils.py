from matplotlib import pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import MSTL
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from statsmodels.tsa.stl._stl import STL


def performMSTL(data, city, on_column="median_count", use_boxcox=False, lambda_bc=None, periods=None):
    df = data.copy()
    df_city = df[df["city"] == city].copy()
    df_city["timestamp"] = pd.to_datetime(df_city["timestamp"])
    df_city = df_city.set_index("timestamp")
    ts = df_city[on_column].asfreq("h")
    ts = ts.interpolate()

    if use_boxcox:
        ts_shifted = ts + 1  # ensure positivity
        if lambda_bc is not None:
            ts_bc = boxcox(ts_shifted.values, lmbda=lambda_bc)
            used_lambda = lambda_bc
        else:
            ts_bc, used_lambda = boxcox(ts_shifted.values)
        print("Used Lambda:", used_lambda)
        ts = pd.Series(ts_bc, index=ts.index, name=f"{on_column}_bc")

    # --- MSTL ---
    res = MSTL(ts, periods=periods).fit()
    return res


def plotMSTLResults(res, city):
    # --- Seasonality plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Daily
    daily_seasonal = res.seasonal['seasonal_24']
    daily_pattern = daily_seasonal.groupby(daily_seasonal.index.hour).mean()
    axes[0].plot(daily_pattern, marker='o')
    axes[0].set_title("Avg Daily Pattern")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Effect")
    axes[0].grid(True)

    # Weekly seasonality as hour-of-week (0–167)
    weekly_seasonal = res.seasonal['seasonal_168']
    df_weekly = pd.DataFrame({
        "weekday": weekly_seasonal.index.dayofweek,  # 0=Mon
        "hour": weekly_seasonal.index.hour,
        "seasonal": weekly_seasonal.values
    })
    weekly_pattern = (
        df_weekly
        .groupby(["weekday", "hour"])["seasonal"]
        .mean()
        .unstack(level=0)
    )
    weekly_pattern_flat = weekly_pattern.values.flatten(order="F")
    axes[1].plot(
        range(168),
        weekly_pattern_flat,
        marker="o",
        linewidth=1.5
    )
    axes[1].set_title("Avg Weekly Pattern (Hour of Week)")
    axes[1].set_xlabel("Hour of Week (0–167)")
    axes[1].set_ylabel("Seasonal Effect")
    axes[1].set_xticks(range(0, 168, 24))
    axes[1].grid(True)

    # Monthly
    yearly_seasonal = res.seasonal['seasonal_8766']
    yearly_pattern = yearly_seasonal.groupby(yearly_seasonal.index.month).mean()
    axes[2].plot(range(1, 13), yearly_pattern, marker='o')
    axes[2].set_xticks(range(1, 13))
    axes[2].set_title("Avg Monthly Pattern")

    fig.suptitle(f"Seasonal Components – {city}", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- Trend and Residuals ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(res.trend)
    axes[0].set_title("Trend")
    axes[0].grid(True)

    axes[1].plot(res.resid)
    axes[1].set_title("Residuals")
    plt.tight_layout()
    plt.show()

    # --- Autocorrelation ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(res.resid, lags=48, ax=axes[0])
    plot_pacf(res.resid, lags=48, ax=axes[1])
    plt.suptitle(f"Residual Diagnostics – {city}", fontsize=14)
    plt.tight_layout()
    plt.show()


def stl_decompose(df_counter, column, period):
    """
    Performs STL decomposition on a specified column and adds seasonal and trend columns.

    Parameters:
    - df_counter: DataFrame with data indexed by timestamp
    - column: str, column name to decompose
    - period: int, period for STL (e.g., 24 for daily, 168 for weekly)

    Returns:
    - df_counter with new columns: seasonal_<period>, trend_<period>, residual_<period>
    """
    stl = STL(df_counter[column], period=period, robust=True)
    res = stl.fit()
    df_counter[f"seasonal_{period}"] = res.seasonal
    df_counter[f"trend_{period}"] = res.trend
    df_counter[f"residual_{period}"] = res.resid
    return df_counter


def process_city_stl(df, city_name, output_csv="../data/stl_results.csv"):
    """
    Processes all counters in a city with daily + weekly STL applied to a specified column.

    Parameters:
    - df: full dataset
    - city_name: str, city to process
    - counter_column: str, column to apply STL to
    - output_csv: path to save results
    """
    df_city = df[df["city"] == city_name].copy()
    df_city["timestamp"] = pd.to_datetime(df_city["timestamp"], utc=True)

    counters = df_city["counter_site_id"].unique()

    first_write = True

    for counter in counters:
        df_counter = df_city[df_city["counter_site_id"] == counter].copy()
        df_counter = df_counter.set_index("timestamp").sort_index()

        # Interpolate missing values in the chosen counter column
        n_nan = df_counter["count"].isna().sum()
        if n_nan > 0:
            df_counter["count"] = df_counter["count"].interpolate(method="time", limit_direction="both")
            print(f"Interpolated {n_nan} NaN values in 'count' for counter {counter}")

        # Log-transform
        df_counter["log_count"] = np.log1p(df_counter["count"])

        # Daily STL
        df_counter = stl_decompose(df_counter, "log_count", period=24)
        df_counter["log_count_deseasonalized"] = df_counter["log_count"] - df_counter["seasonal_24"]

        # Weekly STL
        df_counter = stl_decompose(df_counter, "log_count_deseasonalized", period=168)
        df_counter["log_count_deseasonalized_week"] = df_counter["log_count_deseasonalized"] - df_counter[
            "seasonal_168"]

        # remove trend
        df_counter["log_count_deseason_detrend"] = df_counter["log_count_deseasonalized_week"] - df_counter["trend_24"]
        # convert back from log
        df_counter["count_deseason_detrend"] = np.expm1(df_counter["log_count_deseason_detrend"])

        # Columns to keep
        cols_to_keep = [
            "city",
            "counter_site_id",
            "count",
            "count_deseason_detrend",
            "log_count_deseason_detrend",
            "log_count",
            "trend_24",
            "seasonal_24",
            "residual_24",
            "trend_168",
            "seasonal_168",
            "residual_168",
            "log_count_deseasonalized",
            "log_count_deseasonalized_week",
            "site_rain_accumulation",
            "site_snow_accumulation",
            "longitude",
            "latitude"
        ]

        df_out = df_counter[cols_to_keep].copy()

        # Write to CSV immediately
        df_out.to_csv(output_csv, mode='a', header=first_write, index_label="timestamp")
        first_write = False
        print(f"Processed counter {counter} in city '{city_name}'")


def process_city_mstl(df, city_name, output_csv="../data/mstl_results.csv"):
    df_city = df[df["city"] == city_name].copy()
    df_city["timestamp"] = pd.to_datetime(df_city["timestamp"], utc=True)

    counters = df_city["counter_site"].unique()

    first_write = True

    for counter in counters:
        df_counter = df_city[df_city["counter_site"] == counter].copy()
        df_counter = df_counter.set_index("timestamp").sort_index()
        df_counter["log_count"] = np.log1p(df_counter["count"])
        res = MSTL(df_counter["log_count"], periods=[24, 168]).fit()#, 8766]).fit()
        df_counter["trend"] = res.trend
        df_counter["seasonal_24"] = res.seasonal["seasonal_24"]
        df_counter["seasonal_168"] = res.seasonal["seasonal_168"]
        #df_counter["seasonal_8766"] = res.seasonal["seasonal_8766"]
        df_counter["residual"] = res.resid
        df_counter.to_csv(output_csv, mode='a', header=first_write, index_label="timestamp")
        first_write = False
        print(f"Processed counter {counter} in city '{city_name}'")