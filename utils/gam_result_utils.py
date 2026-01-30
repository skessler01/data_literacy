import numpy as np
import pickle

# Utils for plotting gam results. This file contains methods to extract temperature and rain curve (based on predicting
# from the model for fixed conditions (defined in model parameter))
# In addtion to the curves, the histogram values are computed

# Computes temperature curves. Expects weather to be normalized.
def compute_temp_effect_curves(
        city_models,
        source,
        temp_col=2,
        rain_col=3,
        snow_col=4,
        temp_min=-10,
        temp_max=35,
        rain_levels=(0, 1.0, 2.0),
        n_temps=30,
):
    # compute global temperature mean from city means
    global_mean_temp_real = np.mean([
        mean
        for (_, mean, _) in (
            data["city_means_stds"][0]
            for data in city_models[source].values()
        )
    ])

    # Real temperatures (for labels / plotting)
    temps_real = np.linspace(temp_min, temp_max, n_temps)
    curves = {r_real: [] for r_real in rain_levels}
    snow_fixed = 0.0
    # loop over cities
    for city, data in city_models[source].items():
        # extract mean and std of this city for temp, rain, snow
        gam = data["gam"]
        X_city = data["X"]
        mean_stds = data["city_means_stds"]
        col_temp, mean_temp, std_temp = mean_stds[0]
        col_rain, mean_rain, std_rain = mean_stds[1]
        col_snow, mean_snow, std_snow = mean_stds[2]
        # compute city-wide normalized mean temperature for baseline
        global_mean_temp_norm = (global_mean_temp_real - mean_temp) / std_temp
        # compute normalized rain levels for prediction
        rain_levels_norm = {
            r_real: (r_real - mean_rain) / std_rain
            for r_real in rain_levels
        }
        # Normalized temperatures (for GAM prediction)
        # this has to be done in order to know the temperatures at which we evaluate
        temps_norm = (temps_real - mean_temp) / std_temp

        # set baseline (counter to mean, temperature to avg, rain and snow to 0)
        X_base = X_city.mean(axis=0)
        X_base[temp_col] = global_mean_temp_norm
        X_base[rain_col] = 0.0
        X_base[snow_col] = snow_fixed
        # prepare matrix
        X_base_mat = gam._modelmat(np.tile(X_base, (len(temps_norm), 1)))
        # compute curves based on (normalized) rain level
        for r_real, r_norm in rain_levels_norm.items():
            X_ref = np.tile(X_base, (len(temps_norm), 1))
            X_ref[:, temp_col] = temps_norm
            X_ref[:, rain_col] = r_norm
            X_ref_mat = gam._modelmat(X_ref)
            # predict for these conditions
            delta_eta = (X_ref_mat - X_base_mat) @ gam.coef_
            # convert to percentage value relative to baseline
            curves[r_real].append((np.exp(delta_eta) - 1) * 100)
    out = {}
    # save results
    for r in rain_levels:
        arr = np.vstack(curves[r])
        out[r] = {
            "mean": arr.mean(axis=0),
            "low": np.percentile(arr, 10, axis=0),
            "high": np.percentile(arr, 90, axis=0),
        }

    return {
        "temps": temps_real,
        "global_mean_temp": global_mean_temp_real,
        "curves": out,
    }

# Computes the values displayable in a histogram for temperature distribution
def compute_temp_histograms(
        city_models,
        temp_col=2,
        temp_min=-10,
        temp_max=35,
        n_bins=30,
):
    # Observed
    all_observed = np.hstack([
        np.atleast_2d(v["X"])[:, temp_col] * v["city_means_stds"][0][2] + v["city_means_stds"][0][1]
        for v in city_models["observed"].values()
    ])

    all_forecast = np.hstack([
        np.atleast_2d(v["X"])[:, temp_col] * v["city_means_stds"][0][2] + v["city_means_stds"][0][1]
        for v in city_models["forecast"].values()
    ])

    bins = np.linspace(temp_min, temp_max, n_bins)
    counts_observed, _ = np.histogram(all_observed, bins=bins)
    counts_forecast, _ = np.histogram(all_forecast, bins=bins)

    return {
        "bins": bins,
        "frac_observed": counts_observed / counts_observed.sum(),
        "frac_forecast": counts_forecast / counts_forecast.sum(),
    }

# Takes a dict and computes curve and histogram data and saves it into a pkl file
def save_temp_effect_results(
        city_models,
        out_path,
        temp_col=2,
        rain_col=3,
        snow_col=4,
        temp_min=-10,
        temp_max=35,
        rain_levels=(0, 1.0, 2.0),
):
    # Expected structure of city_models:## city_models = {
    #     "actual": {
    #         "<city_name>": {
    #             "gam": <trained GAM model>,
    #             "X": np.ndarray of shape (n_samples, n_features),
    #             "y": np.ndarray of shape (n_samples,)   # optional for plotting
    #         },
    #         ...
    #     },
    #     "predicted": {
    #         "<city_name>": {
    #             "gam": <trained GAM model>,
    #             "X": np.ndarray of shape (n_samples, n_features),
    #             "y": np.ndarray of shape (n_samples,)
    # optional for plotting
    #         },
    #         ...
    #     }
    # }
    results = {
        "observed": compute_temp_effect_curves(
            city_models,
            source="observed",
            temp_col=temp_col,
            rain_col=rain_col,
            snow_col=snow_col,
            temp_min=temp_min,
            temp_max=temp_max,
            rain_levels=rain_levels,
        ),
        "forecast": compute_temp_effect_curves(
            city_models,
            source="forecast",
            temp_col=temp_col,
            rain_col=rain_col,
            snow_col=snow_col,
            temp_min=temp_min,
            temp_max=temp_max,
            rain_levels=rain_levels,
        ),
        "hist": compute_temp_histograms(
            city_models,
            temp_col=temp_col,
            temp_min=temp_min,
            temp_max=temp_max,
        ),
        "meta": {
            "temp_col": temp_col,
            "rain_col": rain_col,
            "snow_col": snow_col,
            "rain_levels": rain_levels,
            "temp_min": temp_min,
            "temp_max": temp_max,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved temperature effect results to {out_path}")

# Conversion from log diff to percent
def logdiff_to_percent(delta):
    return (np.exp(delta) - 1) * 100

# Computes rain curves. Expects weather to be normalized.
def compute_rain_effect_curves(
    city_models,
    source,
    temps_fixed=(5, 15, 25),
    temp_col=2,
    rain_col=3,
    snow_col=4,
    n_grid=50,
    rain_clip_quantile=99,
):
    models = city_models[source]

    # Compute rain clip in real units across cities
    all_rain_real = np.hstack([
        np.atleast_2d(v["X"])[:, rain_col] * v["city_means_stds"][1][2] + v["city_means_stds"][1][1]
        for v in models.values()
    ])
    rain_clip = np.percentile(all_rain_real, rain_clip_quantile)
    rain_grid_real = np.linspace(0, rain_clip, n_grid)

    snow_fixed = 0.0
    city_curves_by_temp = {T: [] for T in temps_fixed}

    for city, data in models.items():
        X_city = data["X"]
        gam = data["gam"]

        # city-specific mean & std
        col_temp, mean_temp, std_temp = data["city_means_stds"][0]
        col_rain, mean_rain, std_rain = data["city_means_stds"][1]
        col_snow, mean_snow, std_snow = data["city_means_stds"][2]

        rain_grid_norm = (rain_grid_real - mean_rain) / std_rain

        # base row: mean covariates
        X_base = X_city.mean(axis=0)
        X_base[snow_col] = snow_fixed

        for T in temps_fixed:
            # normalize temperature for this city
            T_norm = (T - mean_temp) / std_temp
            # baseline: rain = 0
            X_base_temp = X_base.copy()
            X_base_temp[temp_col] = T_norm
            X_base_temp[rain_col] = 0.0

            X_base_mat = gam._modelmat(
                np.tile(X_base_temp, (len(rain_grid_norm), 1))
            )

            # varying rain
            X_ref = np.tile(X_base, (len(rain_grid_norm), 1))
            X_ref[:, temp_col] = T_norm
            X_ref[:, rain_col] = rain_grid_norm
            X_ref[:, snow_col] = snow_fixed

            X_ref_mat = gam._modelmat(X_ref)

            delta = (X_ref_mat - X_base_mat) @ gam.coef_
            delta_pct = logdiff_to_percent(delta)

            city_curves_by_temp[T].append(delta_pct)

    # aggregate across cities
    summary = {}
    for T in temps_fixed:
        curves = np.vstack(city_curves_by_temp[T])
        summary[T] = {
            "mean": curves.mean(axis=0),
            "low":  np.percentile(curves, 10, axis=0),
            "high": np.percentile(curves, 90, axis=0),
        }
    return {
        "rain_grid": rain_grid_real,
        "rain_clip": rain_clip,
        "temps_fixed": tuple(temps_fixed),
        "curves": summary,
    }

# Computes the values displayable in a histogram for rain distribution
def compute_rain_histograms(
    city_models,
    rain_col=3,
    rain_clip=None,
    n_bins=25,
):

    all_observed = np.hstack([
        np.atleast_2d(v["X"])[:, rain_col] * v["city_means_stds"][1][2] + v["city_means_stds"][1][1]
        for v in city_models["observed"].values()
    ])

    all_forecast = np.hstack([
        np.atleast_2d(v["X"])[:, rain_col] * v["city_means_stds"][1][2] + v["city_means_stds"][1][1]
        for v in city_models["forecast"].values()
    ])

    #all_observed = np.vstack([v["X"] for v in city_models["observed"].values()])[:, rain_col]
    #all_forecast   = np.vstack([v["X"] for v in city_models["forecast"].values()])[:, rain_col]

    if rain_clip is None:
        rain_clip = max(all_observed.max(), all_forecast.max())

    bins = np.linspace(0, rain_clip, n_bins)
    counts_observed, _ = np.histogram(all_observed, bins=bins)
    counts_forecast, _   = np.histogram(all_forecast, bins=bins)

    return {
        "bins": bins,
        "frac_observed": counts_observed / counts_observed.sum(),
        "frac_forecast":   counts_forecast / counts_forecast.sum(),
    }

# Takes a dict and computes curve and histogram data and saves it into a pkl file
def save_rain_effect_results(
    city_models,
    out_path,
    temps_fixed=(5, 15, 25),
    temp_col=2,
    rain_col=3,
    snow_col=4,
    n_bins=25,
):
    observed = compute_rain_effect_curves(
        city_models,
        source="observed",
        temps_fixed=temps_fixed,
        temp_col=temp_col,
        rain_col=rain_col,
        snow_col=snow_col,
    )

    forecast = compute_rain_effect_curves(
        city_models,
        source="forecast",
        temps_fixed=temps_fixed,
        temp_col=temp_col,
        rain_col=rain_col,
        snow_col=snow_col,
    )

    hist = compute_rain_histograms(
        city_models,
        rain_col=rain_col,
        rain_clip=observed["rain_clip"],
        n_bins=n_bins,
    )

    results = {
        "observed": observed,
        "forecast": forecast,
        "hist": hist,
        "meta": {
            "temps_fixed": temps_fixed,
            "temp_col": temp_col,
            "rain_col": rain_col,
            "snow_col": snow_col,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved rain effect results to {out_path}")
