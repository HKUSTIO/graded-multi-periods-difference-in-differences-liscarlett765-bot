from __future__ import annotations

import math

import numpy as np
import pandas as pd


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def generate_panel_data(config: dict, heterogeneous_trend: bool) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_population"]))
    n_units = int(config["n_units"])
    t0 = int(config["pre_periods"])
    t_total = t0 + int(config["post_periods"])
    times = np.arange(1, t_total + 1)

    x_draw = rng.uniform(0.0, 1.0, size=n_units)
    x1 = (x_draw >= 0.3).astype(int)
    x2 = (x_draw >= 0.7).astype(int)
    u = rng.uniform(0.0, 1.0, size=n_units)

    alpha0 = float(config["adoption_intercept"])
    alpha1 = float(config["adoption_slope_middle"])
    alpha2 = float(config["adoption_slope_late"])
    p1 = logistic(alpha0)
    p2 = np.array([logistic(alpha0 + alpha1 * value) for value in x1])
    p3 = np.array([logistic(alpha0 + alpha1 * value) for value in x1 + x2])
    p4 = np.array([logistic(alpha0 + alpha2 * value) for value in x1 + x2])

    cohort = np.zeros(n_units, dtype=int)
    cohort[u <= p1] = t0 + 1
    cohort[(u > p1) & (u <= p2)] = t0 + 2
    cohort[(u > p2) & (u <= p3)] = t0 + 3
    cohort[(u > p3) & (u <= p4)] = t0 + 4

    individual_effect = rng.normal(0.0, float(config["individual_sd"]), size=n_units)
    tau_i = rng.normal(float(config["tau_mean"]), float(config["tau_sd"]), size=n_units)
    time_shocks = rng.uniform(float(config["tau_time_low"]), float(config["tau_time_high"]), size=t_total + 1)

    rows = []
    for time in times:
        if heterogeneous_trend:
            trend_cfg = config["heterogeneous_trend"]
            trend = (
                (time / t_total) * float(trend_cfg["baseline_slope"]) * (1 - x1 - x2)
                + (time / t_total) * float(trend_cfg["x1_slope"]) * x1
                + (time / t_total) * float(trend_cfg["x2_slope"]) * x2
            )
        else:
            trend = np.full(n_units, time / t_total)

        error = rng.normal(0.0, float(config["error_sd"]), size=n_units)
        treated_ever = (cohort > 0).astype(int)
        y0 = float(config["base_level"]) + treated_ever * (-individual_effect) + (1 - treated_ever) * individual_effect + trend + error

        d = ((cohort > 0) & (time >= cohort)).astype(int)
        multiplier = np.zeros(n_units)
        multiplier[cohort == t0 + 1] = 1.0
        multiplier[cohort == t0 + 2] = -2.5
        multiplier[cohort == t0 + 3] = -1.75
        multiplier[cohort == t0 + 4] = -1.0
        tau_it = time_shocks[time] * np.abs(tau_i) * multiplier
        y = y0 + d * tau_it
        relative_time = np.where(cohort > 0, time - cohort, 0)

        for idx in range(n_units):
            rows.append(
                {
                    "id": idx + 1,
                    "x1": int(x1[idx]),
                    "x2": int(x2[idx]),
                    "cohort": int(cohort[idx]),
                    "time": int(time),
                    "relative_time": int(relative_time[idx]),
                    "d": int(d[idx]),
                    "y0": float(y0[idx]),
                    "tau_it": float(tau_it[idx]),
                    "y": float(y[idx]),
                }
            )

    return pd.DataFrame(rows, columns=["id", "x1", "x2", "cohort", "time", "relative_time", "d", "y0", "tau_it", "y"])


def summarize_group_shares_and_att(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per treated cohort and one row for all treated observations.
    """
    unit_cohort = data.drop_duplicates(subset=["id"])[["id", "cohort"]]
    n_units = unit_cohort.shape[0]
    n_rows = data.shape[0]

    treated_cohorts = sorted(int(g) for g in unit_cohort["cohort"].unique() if g > 0)

    rows = []
    for g in treated_cohorts:
        fraction = float((unit_cohort["cohort"] == g).sum()) / float(n_units)
        treated_mask = (data["cohort"] == g) & (data["d"] == 1)
        att = float(data.loc[treated_mask, "tau_it"].mean()) if treated_mask.any() else float("nan")
        rows.append({"group": f"cohort_{g}", "fraction": fraction, "att": att})

    all_treated_mask = data["d"] == 1
    fraction_all = float(all_treated_mask.sum()) / float(n_rows)
    att_all = float(data.loc[all_treated_mask, "tau_it"].mean()) if all_treated_mask.any() else float("nan")
    rows.append({"group": "all_treated", "fraction": fraction_all, "att": att_all})

    return pd.DataFrame(rows, columns=["group", "fraction", "att"])


def estimate_cohort_did(data: pd.DataFrame, cohort: int, event_time: int, control_group: str) -> float:
    """
    Return a two-period DID estimate for one treatment cohort and event time.
    """
    target_period = cohort + event_time
    baseline_period = cohort - 1

    treated = data[data["cohort"] == cohort]
    y_treated_t = treated.loc[treated["time"] == target_period, "y"].mean()
    y_treated_b = treated.loc[treated["time"] == baseline_period, "y"].mean()

    if control_group == "never":
        control = data[data["cohort"] == 0]
    elif control_group == "notyet":
        control = data[(data["cohort"] == 0) | (data["cohort"] > target_period)]
    else:
        raise ValueError(f"Unknown control_group: {control_group}")

    y_ctrl_t = control.loc[control["time"] == target_period, "y"].mean()
    y_ctrl_b = control.loc[control["time"] == baseline_period, "y"].mean()

    return float((y_treated_t - y_treated_b) - (y_ctrl_t - y_ctrl_b))


def estimate_event_study(data: pd.DataFrame, event_times: list[int], control_group: str) -> pd.DataFrame:
    """
    Return cohort-event DID estimates.
    """
    treated_cohorts = sorted(int(g) for g in data["cohort"].unique() if g > 0)
    min_time = int(data["time"].min())
    max_time = int(data["time"].max())

    rows = []
    for g in treated_cohorts:
        baseline = g - 1
        if baseline < min_time or baseline > max_time:
            continue
        for e in event_times:
            target = g + int(e)
            if target < min_time or target > max_time:
                continue
            estimate = estimate_cohort_did(data, cohort=g, event_time=int(e), control_group=control_group)
            rows.append({"cohort": g, "event_time": int(e), "estimate": estimate})

    result = pd.DataFrame(rows, columns=["cohort", "event_time", "estimate"])
    if not result.empty:
        result = result.sort_values(["cohort", "event_time"]).reset_index(drop=True)
    return result


def aggregate_post_treatment_effects(event_study: pd.DataFrame) -> float:
    """
    Return the average estimate over post-treatment event times.
    """
    post = event_study[event_study["event_time"] >= 0]
    return float(post["estimate"].mean())


def estimate_twfe_coefficient(data: pd.DataFrame) -> float:
    """
    Return the coefficient from a residualized two-way fixed effects regression of y on d.
    """
    y = data["y"].astype(float)
    d = data["d"].astype(float)

    y_unit = data.groupby("id")["y"].transform("mean")
    y_time = data.groupby("time")["y"].transform("mean")
    y_grand = y.mean()
    y_ddot = y - y_unit - y_time + y_grand

    d_unit = data.groupby("id")["d"].transform("mean")
    d_time = data.groupby("time")["d"].transform("mean")
    d_grand = d.mean()
    d_ddot = d - d_unit - d_time + d_grand

    numerator = float((d_ddot * y_ddot).sum())
    denominator = float((d_ddot * d_ddot).sum())
    return numerator / denominator
