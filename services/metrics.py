"""
metrics.py
~~~~~~~~~~
Shared numerical metrics for HydroVision services.

Functions
---------
rmse(actual, predicted)          Root-mean-square error (same units as data).
mape(actual, predicted)          Mean absolute percentage error (%), zero-safe.
mape_grade(mape_val)             (key, label) accuracy tier from MAPE value.
mann_kendall(x)                  Full Mann-Kendall monotonic trend test result.

All functions accept array-like inputs and return Python scalars or dicts.
They never raise on NaN-heavy inputs — they return float('nan') or a safe
sentinel instead, so callers don't need to guard against empty slices.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import scipy.stats


# ── RMSE ──────────────────────────────────────────────────────────────────────

def rmse(actual: Any, predicted: Any) -> float:
    """
    Root-mean-square error between *actual* and *predicted*.

    NaN pairs are excluded from the calculation.  Returns ``float('nan')``
    when fewer than two valid pairs remain.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 2:
        return float('nan')
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))


# ── MAPE ──────────────────────────────────────────────────────────────────────

def mape(actual: Any, predicted: Any, min_abs_threshold: float | None = None) -> float:
    """
    Mean absolute percentage error (%) between *actual* and *predicted*.

    Zero-safe: pairs where ``|actual| < min_abs_threshold`` are excluded to
    avoid division-by-zero inflation.  The threshold defaults to 1 % of the
    mean of |actual| when not supplied.

    Returns ``float('nan')`` when no valid pairs remain or all actuals are
    near zero.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) < 2:
        return float('nan')

    threshold = min_abs_threshold if min_abs_threshold is not None else float(np.mean(np.abs(a))) * 0.01
    nonzero = np.abs(a) > threshold
    if nonzero.sum() < 2:
        return float('nan')
    return float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)


# ── MAPE grade ────────────────────────────────────────────────────────────────

def mape_grade(mape_val: float) -> Tuple[str, str]:
    """
    Return ``(key, label)`` accuracy tier for a MAPE value.

    Tiers
    -----
    strong         MAPE < 10 %
    moderate       10 % ≤ MAPE < 20 %
    poor           20 % ≤ MAPE ≤ 500 %
    indeterminate  NaN or MAPE > 500 % (model has not converged or no data)

    The *key* is suitable for CSS class names; *label* is display text.
    """
    if math.isnan(mape_val) or mape_val > 500:
        return ('indeterminate', 'Indeterminate')
    if mape_val < 10:
        return ('strong', 'Strong')
    if mape_val < 20:
        return ('moderate', 'Moderate')
    return ('poor', 'Poor')


# ── Mann-Kendall trend test ────────────────────────────────────────────────────

def mann_kendall(x: Any) -> Dict[str, Any]:
    """
    Two-sided Mann-Kendall monotonic trend test.

    Parameters
    ----------
    x : array-like
        1-D sequence of observations in time order.  NaNs are dropped before
        the test so the caller does not need to pre-filter.

    Returns
    -------
    dict with keys:
        n           int    — number of valid observations used.
        s           float  — Mann-Kendall S statistic.
        var_s       float  — variance of S under H₀.
        z           float  — standardised test statistic.
        p           float  — two-tailed p-value.
        tau         float  — Kendall's τ (normalised correlation, −1 to 1).
        significant bool   — True when p < 0.05.
        trend       str    — 'Increasing ↑', 'Decreasing ↓', or 'No trend'.
        slope       float  — Theil-Sen median slope (units per observation).

    All float fields are ``float('nan')`` when fewer than 4 valid points exist.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    nan_result: Dict[str, Any] = dict(
        n=n, s=float('nan'), var_s=float('nan'), z=float('nan'),
        p=float('nan'), tau=float('nan'), significant=False,
        trend='Too short', slope=float('nan'),
    )
    if n < 4:
        return nan_result

    # S statistic
    s = 0.0
    for i in range(n - 1):
        s += float(np.sum(np.sign(arr[i + 1:] - arr[i])))

    # Variance under H₀ (no tie correction — adequate for continuous data)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    # Continuity-corrected Z
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = float(2 * (1 - scipy.stats.norm.cdf(abs(z))))

    # Kendall's τ
    tau = s / (0.5 * n * (n - 1))

    # Theil-Sen slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((arr[j] - arr[i]) / (j - i))
    slope = float(np.median(slopes)) if slopes else float('nan')

    significant = p < 0.05
    if significant:
        trend = 'Increasing ↑' if s > 0 else 'Decreasing ↓'
    else:
        trend = 'No trend'

    return dict(
        n=n, s=s, var_s=var_s, z=z, p=p, tau=tau,
        significant=significant, trend=trend, slope=slope,
    )
