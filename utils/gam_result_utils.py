import numpy as np
import pickle


def compute_temp_effect_curves(
        city_models,
        source,
        temp_col=2,
        rain_col=3,
        snow_col=4,
        temp_min=-10,
        temp_max=35,
        rain_levels=(0, 1.0, 3.0),
        n_temps=30,
):
    all_X = np.vstack([v["X"] for v in city_models[source].values()])
    global_mean_temp = all_X[:, temp_col].mean()
    temps = np.linspace(temp_min, temp_max, n_temps)

    curves = {r: [] for r in rain_levels}
    snow_fixed = 0.0

    for city, data in city_models[source].items():
        gam = data["gam"]
        X_city = data["X"]

        X_base = X_city.mean(axis=0)
        X_base[temp_col] = global_mean_temp
        X_base[rain_col] = 0.0
        X_base[snow_col] = snow_fixed

        X_base_mat = gam._modelmat(np.tile(X_base, (len(temps), 1)))

        for r in rain_levels:
            X_ref = np.tile(X_base, (len(temps), 1))
            X_ref[:, temp_col] = temps
            X_ref[:, rain_col] = r

            X_ref_mat = gam._modelmat(X_ref)
            delta_eta = (X_ref_mat - X_base_mat) @ gam.coef_
            curves[r].append((np.exp(delta_eta) - 1) * 100)

    out = {}
    for r in rain_levels:
        arr = np.vstack(curves[r])
        out[r] = {
            "mean": arr.mean(axis=0),
            "low": np.percentile(arr, 10, axis=0),
            "high": np.percentile(arr, 90, axis=0),
        }

    return {
        "temps": temps,
        "global_mean_temp": global_mean_temp,
        "curves": out,
    }


def compute_temp_histograms(
        city_models,
        temp_col=2,
        temp_min=-10,
        temp_max=35,
        n_bins=30,
):
    all_actual = np.vstack([v["X"] for v in city_models["actual"].values()])[:, temp_col]
    all_pred = np.vstack([v["X"] for v in city_models["predicted"].values()])[:, temp_col]

    bins = np.linspace(temp_min, temp_max, n_bins)
    counts_actual, _ = np.histogram(all_actual, bins=bins)
    counts_pred, _ = np.histogram(all_pred, bins=bins)

    return {
        "bins": bins,
        "frac_actual": counts_actual / counts_actual.sum(),
        "frac_pred": counts_pred / counts_pred.sum(),
    }


def save_temp_effect_results(
        city_models,
        out_path,
        temp_col=2,
        rain_col=3,
        snow_col=4,
        temp_min=-10,
        temp_max=35,
        rain_levels=(0, 1.0, 3.0),
):
    results = {
        "actual": compute_temp_effect_curves(
            city_models,
            source="actual",
            temp_col=temp_col,
            rain_col=rain_col,
            snow_col=snow_col,
            temp_min=temp_min,
            temp_max=temp_max,
            rain_levels=rain_levels,
        ),
        "predicted": compute_temp_effect_curves(
            city_models,
            source="predicted",
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


import numpy as np


def logdiff_to_percent(delta):
    return (np.exp(delta) - 1) * 100


def compute_rain_effect_curves(
    city_models,
    source,
    temps_fixed=(5, 20, 35),
    temp_col=2,
    rain_col=3,
    snow_col=4,
    n_grid=50,
    rain_clip_quantile=99,
):
    models = city_models[source]

    # rain grid
    all_rain = np.vstack([v["X"] for v in models.values()])[:, rain_col]
    rain_clip = np.percentile(all_rain, rain_clip_quantile)
    rain_grid = np.linspace(0, rain_clip, n_grid)

    snow_fixed = 0.0
    city_curves_by_temp = {T: [] for T in temps_fixed}

    for city, data in models.items():
        X_city = data["X"]
        gam = data["gam"]

        # base row: mean covariates
        X_base = X_city.mean(axis=0)
        X_base[snow_col] = snow_fixed

        for T in temps_fixed:
            # baseline: rain = 0
            X_base_temp = X_base.copy()
            X_base_temp[temp_col] = T
            X_base_temp[rain_col] = 0.0

            X_base_mat = gam._modelmat(
                np.tile(X_base_temp, (len(rain_grid), 1))
            )

            # varying rain
            X_ref = np.tile(X_base, (len(rain_grid), 1))
            X_ref[:, temp_col] = T
            X_ref[:, rain_col] = rain_grid
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
        "rain_grid": rain_grid,
        "rain_clip": rain_clip,
        "temps_fixed": tuple(temps_fixed),
        "curves": summary,
    }


def compute_rain_histograms(
    city_models,
    rain_col=3,
    rain_clip=None,
    n_bins=25,
):
    all_actual = np.vstack([v["X"] for v in city_models["actual"].values()])[:, rain_col]
    all_pred   = np.vstack([v["X"] for v in city_models["predicted"].values()])[:, rain_col]

    if rain_clip is None:
        rain_clip = max(all_actual.max(), all_pred.max())

    bins = np.linspace(0, rain_clip, n_bins)
    counts_actual, _ = np.histogram(all_actual, bins=bins)
    counts_pred, _   = np.histogram(all_pred, bins=bins)

    return {
        "bins": bins,
        "frac_actual": counts_actual / counts_actual.sum(),
        "frac_pred":   counts_pred / counts_pred.sum(),
    }


import pickle


def save_rain_effect_results(
    city_models,
    out_path,
    temps_fixed=(5, 20, 35),
    temp_col=2,
    rain_col=3,
    snow_col=4,
    n_bins=25,
):
    actual = compute_rain_effect_curves(
        city_models,
        source="actual",
        temps_fixed=temps_fixed,
        temp_col=temp_col,
        rain_col=rain_col,
        snow_col=snow_col,
    )

    predicted = compute_rain_effect_curves(
        city_models,
        source="predicted",
        temps_fixed=temps_fixed,
        temp_col=temp_col,
        rain_col=rain_col,
        snow_col=snow_col,
    )

    hist = compute_rain_histograms(
        city_models,
        rain_col=rain_col,
        rain_clip=actual["rain_clip"],
        n_bins=n_bins,
    )

    results = {
        "actual": actual,
        "predicted": predicted,
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
