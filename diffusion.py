"""
Exact port of Excel LAMBDA diffusion functions.
Matches: Diffusion_L, Type1_L, Type2_L, StepFunction_L, TimeIndex_L
"""
import numpy as np
from datetime import datetime


def time_index(dt):
    """TimeIndex_L: Convert date to decimal year."""
    if isinstance(dt, (int, float)):
        # Already a number (Excel serial date) - convert
        from datetime import timedelta
        base = datetime(1899, 12, 30)
        dt = base + timedelta(days=dt)
    if dt is None:
        return 0
    return dt.year + (dt.month - 1) / 12 + (dt.day - 1) / 365


def step_function(start_val, end_val, launch_ti, time_ti, interval):
    """StepFunction_L: Binary step transition at launch date."""
    if time_ti + interval <= launch_ti:
        return start_val * interval
    elif time_ti >= launch_ti:
        return end_val * interval
    else:
        weight = (launch_ti - time_ti) / interval
        return weight * start_val + (1 - weight) * end_val


def type1(dbl_max, dbl_tmax, dbl_lag, v_launch, v_time, dbl_interval):
    """
    Type1_L: Logistic S-curve diffusion.
    Uses 15%/97% thresholds to derive con and slo parameters.
    """
    launch_ti = time_index(v_launch)
    time_ti = time_index(v_time)

    if dbl_tmax == dbl_lag or dbl_tmax == 0:
        if dbl_tmax < 0:
            return 0
        return step_function(0, dbl_max, launch_ti + dbl_tmax, time_ti, dbl_interval)

    if dbl_interval == 0:
        # Point estimate
        if time_ti < launch_ti:
            return 0
        con = (1 / (1 - dbl_lag / dbl_tmax)) * np.log(
            (1 / 0.15 - 1) / ((1 / 0.97 - 1) ** (dbl_lag / dbl_tmax))
        )
        slo = (1 / dbl_tmax) * (np.log(1 / 0.97 - 1) - con)
        return dbl_max / (1 + np.exp(con + slo * (time_ti - launch_ti)))
    else:
        # Interval average (integral / interval)
        lower = max(time_ti, launch_ti)
        upper = max(time_ti + dbl_interval, launch_ti)
        con = (1 / (1 - dbl_lag / dbl_tmax)) * np.log(
            (1 / 0.15 - 1) / ((1 / 0.97 - 1) ** (dbl_lag / dbl_tmax))
        )
        slo = (1 / dbl_tmax) * (np.log(1 / 0.97 - 1) - con)

        def _integral(t):
            x = con + slo * (t - launch_ti)
            return dbl_max / slo * (x - np.log(1 + np.exp(x)))

        lower_limit = _integral(lower)
        upper_limit = _integral(upper)
        return (upper_limit - lower_limit) / dbl_interval


def type2(dbl_max, dbl_tmax, v_launch, v_time, dbl_interval):
    """
    Type2_L: Exponential saturation diffusion.
    slo = 3.50655789731998 / TMax
    """
    launch_ti = time_index(v_launch)
    time_ti = time_index(v_time)

    if dbl_tmax <= 0:
        return step_function(0, dbl_max, launch_ti, time_ti, dbl_interval)

    if dbl_interval == 0:
        # Point estimate
        if time_ti < launch_ti:
            return 0
        slo = 3.50655789731998 / dbl_tmax
        return dbl_max * (1 - 1 / np.exp(slo * (time_ti - launch_ti)))
    else:
        # Interval average
        lower = max(time_ti, launch_ti)
        upper = max(time_ti + dbl_interval, launch_ti)
        slo = 3.50655789731998 / dbl_tmax

        def _integral(t):
            return dbl_max * (t + (1 / slo) * np.exp(-(slo * (t - launch_ti))))

        lower_limit = _integral(lower)
        upper_limit = _integral(upper)
        return (upper_limit - lower_limit) / dbl_interval


def diffusion_l(dbl_type, dbl_max, dbl_tmax, v_launch, v_time, dbl_interval):
    """
    Diffusion_L: Master dispatcher.
    Blends Type1 (logistic) and Type2 (exponential) based on dbl_type.
    dbl_type=1 -> pure Type1, dbl_type=2 -> pure Type2, in between -> blend.
    Type1 uses lag = TMax/4.
    """
    try:
        dbl_type_adj = min(max(dbl_type, 1), 2) - 1  # 0 for type1, 1 for type2
        if dbl_tmax < 0:
            return 0
        t1 = type1(dbl_max, dbl_tmax, dbl_tmax / 4, v_launch, v_time, dbl_interval)
        t2 = type2(dbl_max, dbl_tmax, v_launch, v_time, dbl_interval)
        return (1 - dbl_type_adj) * t1 + dbl_type_adj * t2
    except Exception:
        return 0


def tot_pts(new_pts, persist):
    """
    TotPts_L: Convolution of new patient cohorts with reversed persistency.
    new_pts: array of new patients per month (up to current month)
    persist: array of persistency rates (M1, M2, ... same length)
    Returns: total patients at current month.
    """
    n = len(new_pts)
    if n == 0:
        return 0
    persist_use = persist[:n]
    # Reverse persistency and dot with new_pts
    return np.dot(new_pts, persist_use[::-1])


def compute_diffusion_curve(dbl_type, peak, start, ct, ttp_years, launch_date, dates):
    """
    Compute a full diffusion curve over a list of dates.
    Returns array of diffusion values.
    The formula from Excel: start + Diffusion_L(ct, peak-start, ttp_years, launch, date, 1/12)
    """
    result = np.zeros(len(dates))
    final_impact = peak - start
    for i, dt in enumerate(dates):
        diff = diffusion_l(ct, final_impact, ttp_years, launch_date, dt, 1/12)
        result[i] = start + diff
    return result
