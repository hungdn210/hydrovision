from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest
from .figure_theme import (
    GRID, dark_layout, axis_style,
    legend_h, MARGIN_SUBPLOT, style_subplot_titles,
    method_note_annotation,
)


def _fallback_scenario_analysis(scenario_result: Dict[str, Any]) -> str:
    """
    Local rule-based analysis for scenario results.
    Always returns useful structured HTML — no external AI required.
    """
    station  = scenario_result.get('station', '').replace('_', ' ')
    target   = scenario_result.get('target_feature', '').replace('_', ' ')
    driver   = scenario_result.get('driver_feature', '').replace('_', ' ')
    scale    = float(scenario_result.get('scale_pct', 0))
    duration = int(scenario_result.get('duration_months', 0))
    unit     = scenario_result.get('unit', '')
    stats    = scenario_result.get('stats', {}) or {}
    sens     = scenario_result.get('sensitivity', {}) or {}

    mean_delta     = float(stats.get('mean_delta', 0))
    max_delta      = float(stats.get('max_delta', 0))
    mean_delta_pct = float(stats.get('mean_delta_pct', 0))
    elasticity     = float(sens.get('elasticity', 0))
    dominant_lag   = int(sens.get('dominant_lag', 0))
    r_value        = float(sens.get('r_value', 0))
    fit_r2         = float(sens.get('fit_r2', 0))
    used_fallback  = bool(sens.get('used_fallback', False))
    model_type     = str(sens.get('model_type', 'statistical'))
    is_direct      = bool(sens.get('direct', False))
    baseline_source = scenario_result.get('baseline_source', 'unknown')

    # ── Interpret direction ────────────────────────────────────────────────────
    driver_direction = 'increase' if scale > 0 else 'decrease'
    response_direction = 'increase' if mean_delta > 0 else 'decrease'
    sign = '+' if scale >= 0 else ''

    # ── Classify elasticity ────────────────────────────────────────────────────
    abs_elast = abs(elasticity)
    if abs_elast > 1.5:
        elast_class = 'highly elastic'
        elast_note  = f'A 1% {driver_direction} in {driver} produces approximately {abs_elast:.2f}% change in {target} — a strong amplified response.'
    elif abs_elast > 0.5:
        elast_class = 'moderately elastic'
        elast_note  = f'A 1% {driver_direction} in {driver} produces approximately {abs_elast:.2f}% change in {target} — a proportional response.'
    else:
        elast_class = 'inelastic'
        elast_note  = f'A 1% {driver_direction} in {driver} produces only {abs_elast:.2f}% change in {target} — a muted, dampened response.'

    # ── Interpret lag ──────────────────────────────────────────────────────────
    if is_direct:
        lag_note = f'{target} responds instantaneously because the driver and target are the same variable (direct scaling applied).'
    elif dominant_lag == 0:
        lag_note = f'The dominant response occurs within the same month as the driver change — near-instantaneous propagation.'
    else:
        lag_note = (
            f'The dominant response lags the driver by {dominant_lag} month(s), meaning the peak effect on {target} '
            f'propagates approximately {dominant_lag} month(s) after the driver change is applied.'
        )

    # ── Classify model confidence ──────────────────────────────────────────────
    if is_direct:
        confidence_note = 'Confidence is not applicable — direct scaling was used (driver equals target variable).'
    elif used_fallback:
        confidence_note = (
            f'The fitted model has a low historical correlation (R={r_value:.2f}, R²={fit_r2:.2f}). '
            f'The relationship between {driver} and {target} is weak — treat this scenario as indicative only.'
        )
    elif r_value >= 0.7:
        confidence_note = (
            f'The model shows a strong historical fit (R={r_value:.2f}, R²={fit_r2:.2f}), '
            f'supporting reasonable confidence in the directional scenario estimate.'
        )
    else:
        confidence_note = (
            f'The model shows a moderate historical fit (R={r_value:.2f}, R²={fit_r2:.2f}). '
            f'Interpret the scenario as a directional estimate, not a precise projection.'
        )

    # ── Baseline source note ───────────────────────────────────────────────────
    if baseline_source == 'trained_model_csv':
        baseline_note = 'The baseline forecast is drawn from a trained ML model CSV.'
    else:
        baseline_note = 'The baseline forecast uses a statistical mean projection (no trained model CSV was available).'

    # ── Unit-safe formatting ───────────────────────────────────────────────────
    unit_str = f' {unit}' if unit else ''
    mean_delta_str = f'{mean_delta:+.3f}{unit_str}'
    max_delta_str  = f'{max_delta:.3f}{unit_str}'

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>A <strong>{sign}{scale:.0f}% {driver_direction}</strong> in <strong>{driver}</strong> '
        f'applied over <strong>{duration} month(s)</strong> at <strong>{station}</strong> is estimated to cause '
        f'a mean <strong>{response_direction} of {mean_delta_str} ({mean_delta_pct:+.1f}%)</strong> in <strong>{target}</strong>. '
        f'The system response is classified as <strong>{elast_class}</strong> (elasticity = {elasticity:.2f}), '
        f'and the dominant response {("is near-instantaneous." if dominant_lag == 0 else f"lags the driver by {dominant_lag} month(s).")} '
        f'{confidence_note}</p>'

        '<p><strong>Driver and Response Quantification</strong></p>'
        f'<p>The {sign}{scale:.0f}% {driver_direction} in {driver} produces a mean {response_direction} '
        f'of {mean_delta_str} ({mean_delta_pct:+.1f}%) in {target} across the {duration}-month shock window. '
        f'The peak single-month impact reaches {max_delta_str}, which represents the largest instantaneous effect within the simulation period. '
        'The difference between mean and peak impact reflects the distributed nature of the lagged response model — '
        'early months may show smaller effects as the driver anomaly propagates through the lag structure, '
        'with the full effect only materialising once all lagged terms have been activated. '
        'This temporal profile is important for operational planning: water managers should anticipate a build-up of impact over the shock period rather than an immediate step change.</p>'

        '<p><strong>Sensitivity and Elasticity</strong></p>'
        f'<p>The elasticity of {elasticity:.2f} indicates that the system is <strong>{elast_class}</strong> with respect to {driver} changes. '
        f'{elast_note} '
        + ('A highly elastic response means that even modest driver anomalies — such as a drought reducing rainfall by 20% — '
           'can produce disproportionately large changes in discharge or water level, compressing or amplifying flood and drought risks significantly. '
           if abs_elast > 1.5 else
           'A moderately elastic response indicates roughly proportional coupling between driver and target — '
           'planning can use the percentage change in the driver as a first-order estimate of the percentage change in the target. '
           if abs_elast > 0.5 else
           'An inelastic response indicates that the target is buffered against driver anomalies — '
           'possibly by catchment storage, groundwater baseflow, or upstream regulation that dampens the direct rainfall-runoff relationship. ')
        + 'This elasticity estimate is derived from the historical covariance structure and should be treated as a basin-specific sensitivity index, not a physical routing coefficient.</p>'

        '<p><strong>Lag Dynamics</strong></p>'
        f'<p>{lag_note} '
        + (f'A {dominant_lag}-month propagation time implies that the {target} response to a driver change in a given month '
           f'peaks approximately {dominant_lag} month(s) later, after the anomaly has travelled through the hydrological pathway '
           '(catchment storage, soil moisture routing, channel transit, and groundwater interaction). '
           'This lag structure is critical for operational lead-time: water managers have a window of approximately '
           f'{dominant_lag} month(s) between observing an anomaly in {driver} and the full impact appearing in {target}.'
           if dominant_lag > 0 else
           f'Near-instantaneous coupling between {driver} and {target} suggests a fast-responding catchment with limited storage buffering — '
           'typical of steep, impermeable, or highly regulated basins where rainfall-runoff conversion is rapid.')
        + '</p>'

        '<p><strong>Model Confidence and Fit Quality</strong></p>'
        f'<p>{confidence_note} '
        f'{baseline_note} '
        'The distributed-lag monthly anomaly model is fitted to the historical record and captures the average linear response of the target to driver anomalies across all seasons and years. '
        'It will not capture non-linear responses (e.g. threshold effects above bankfull discharge), regime changes (e.g. post-dam construction), '
        'or conditions outside the historical driver range. '
        'For driver shocks larger than the historical inter-annual variability, the model should be treated as an extrapolation with increasing uncertainty.</p>'

        '<p><strong>Operational Implications</strong></p>'
        '<ul>'
        f'<li><strong>Flood / drought preparedness:</strong> A {sign}{scale:.0f}% change in {driver} over {duration} month(s) '
        f'translates to a mean {response_direction} of {mean_delta_str} in {target}. '
        'If the driver anomaly is associated with a known climate mode (e.g. El Niño reducing rainfall), '
        'water managers can use this estimate to pre-position reservoir storage, adjust irrigation allocations, or issue early drought warnings.</li>'
        f'<li><strong>Reservoir and irrigation scheduling:</strong> The {dominant_lag}-month lag structure defines the operational lead time available. '
        f'Early detection of a {driver} anomaly of {sign}{scale:.0f}% gives approximately {dominant_lag} month(s) to adjust operations before the full impact on {target} materialises.</li>'
        '<li><strong>Limitations for decision use:</strong> These are statistical scenario responses, not hydraulic simulations. '
        'Pair this analysis with field observations, real-time monitoring, and hydraulic assessments before making high-stakes infrastructure or water-allocation decisions.</li>'
        '</ul>'
    )


def _generate_scenario_analysis(scenario_result: Dict[str, Any]) -> str:
    """
    Generate AI-enhanced scenario analysis using Gemini.
    Falls back to the local rule-based analysis on any failure.
    """
    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    if not api_key or api_key == 'your_google_gemini_api_key_here':
        return _fallback_scenario_analysis(scenario_result)

    station  = scenario_result.get('station', '').replace('_', ' ')
    target   = scenario_result.get('target_feature', '').replace('_', ' ')
    driver   = scenario_result.get('driver_feature', '').replace('_', ' ')
    scale    = scenario_result.get('scale_pct', 0)
    duration = scenario_result.get('duration_months', 0)
    unit     = scenario_result.get('unit', '')
    stats    = scenario_result.get('stats', {}) or {}
    sens     = scenario_result.get('sensitivity', {}) or {}

    prompt = (
        'Act as a professional hydrologist writing a detailed technical interpretation of a hydrological what-if scenario simulation.\n\n'
        'Write the response in markdown and structure it exactly as follows:\n\n'
        '**Executive Summary**\n'
        '4-5 sentences. State the driver variable, the shock applied, the overall direction and magnitude of impact on the target, '
        'the elasticity classification, the dominant lag, and the model confidence level.\n\n'
        '**Driver and Response Quantification**\n'
        'A paragraph (4-5 sentences). Precisely quantify mean and peak impact with units. '
        'Explain why mean and peak differ (lag structure distributing impact over time). '
        'Discuss the temporal build-up of impact and its relevance to operational lead times.\n\n'
        '**Sensitivity and Elasticity**\n'
        'A paragraph (4-5 sentences). Interpret the elasticity value in detail — is the system amplifying, proportional, or dampened? '
        'Discuss what a highly elastic or inelastic response implies about catchment characteristics (storage, regulation, baseflow). '
        'Explain how elasticity should be applied in planning for drought or flood scenarios.\n\n'
        '**Lag Dynamics**\n'
        'A paragraph (3-4 sentences). Interpret the dominant lag operationally. '
        'Explain what physical processes create the lag (soil moisture routing, channel transit, groundwater interaction). '
        'State the operational lead-time window this gives water managers.\n\n'
        '**Model Confidence and Fit Quality**\n'
        'A paragraph (3-4 sentences). Interpret R and R² values — is the historical driver–target relationship strong enough to support planning decisions? '
        'State clearly this is a statistical response model, not a hydraulic simulation. '
        'Identify the main conditions where the model may over- or under-estimate the true impact.\n\n'
        '**Operational Implications**\n'
        'Exactly 3 bullet points:\n'
        '- **Flood/drought preparedness:** how this scenario result should inform emergency preparedness or water security planning.\n'
        '- **Reservoir and irrigation scheduling:** how the lag structure defines the actionable lead-time for operational decisions.\n'
        '- **Decision limitations:** remind the reader to pair this with field observations and hydraulic assessment before high-stakes decisions.\n\n'
        'Rules:\n'
        '- Use professional hydrological language throughout.\n'
        '- Always cite specific numbers.\n'
        '- Replace underscores with spaces.\n'
        '- Make clear this is a statistical response model, not a physical simulation.\n'
        '- No introduction, no sign-off.\n\n'
        f'Station: {station}\n'
        f'Target variable: {target} ({unit})\n'
        f'Driver variable: {driver}\n'
        f'Driver change: {scale:+.0f}%\n'
        f'Duration: {duration} month(s)\n'
        f'Mean impact: {stats.get("mean_delta", 0):+.3f} {unit}\n'
        f'Peak impact: {stats.get("max_delta", 0):.3f} {unit}\n'
        f'Mean % change: {stats.get("mean_delta_pct", 0):+.1f}%\n'
        f'Elasticity: {sens.get("elasticity", 0):.2f}\n'
        f'Dominant lag: {sens.get("dominant_lag", 0)} month(s)\n'
        f'Model fit R: {sens.get("r_value", 0):.2f}\n'
        f'Model fit R²: {sens.get("fit_r2", 0):.2f}\n'
        f'Model type: {sens.get("model_type", "unknown")}\n'
        f'Weak fit warning: {sens.get("used_fallback", False)}\n'
    )

    try:
        raw = _gemini_generate(api_key, prompt)
        if not raw or not raw.strip():
            return _fallback_scenario_analysis(scenario_result)
        return markdown.markdown(raw)
    except Exception:
        return _fallback_scenario_analysis(scenario_result)


class ScenarioService:
    """
    What-If scenario modelling.

    Workflow:
      1. Load historical target and driver series and resample to monthly means.
      2. Remove monthly climatology to form relative anomalies.
      3. Fit a distributed-lag response model with target persistence and
         driver lags across the previous 0..3 months.
      4. Load the baseline forecast from the prediction CSV files, or fall back
         to a simple monthly projection.
      5. Inject a driver shock during the selected months and propagate it
         through the lagged response model to obtain month-varying target impacts.
      6. Return both series + a Plotly figure.
    """

    # Map Mekong internal feature names → prediction folder names
    _MEKONG_FEAT_FOLDER = {
        'Discharge': 'Water_Discharge',
        'Water_Level': 'Water_Level',
        'Rainfall': 'Rainfall',
        'Total_Suspended_Solids': 'Total_Suspended_Solids',
    }
    _MAX_DRIVER_LAGS = 3

    def __init__(self, repository, data_dir: str | Path = 'data') -> None:
        self.repo = repository
        self.data_dir = Path(data_dir)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_repo(self, station: str):
        """Return the DataRepository that owns station, or None."""
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if station in r.station_index), None)
        if station in getattr(self.repo, 'station_index', {}):
            return self.repo
        return None

    def _load_series(self, repo, station: str, feature: str) -> pd.Series:
        fd = repo.station_index[station]['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = repo.get_feature_series(req)
        ts = df.set_index('Timestamp')['Value'].sort_index()
        ts.index = pd.to_datetime(ts.index)
        return ts

    def _mekong_feat_folder(self, feature: str) -> str:
        return self._MEKONG_FEAT_FOLDER.get(feature, feature)

    def _load_csv_forecast(self, repo, station: str, feature: str, model: str, horizon: int) -> pd.Series | None:
        dataset = repo.dataset
        if dataset == 'mekong':
            path = (self.data_dir / 'Mekong' / 'prediction_results' / 'station_predictions_future'
                    / self._mekong_feat_folder(feature) / model / f'{station}.csv')
        elif dataset == 'lamah':
            path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions_future'
                    / model / f'{station}.csv')
            if not path.exists():
                path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions_future'
                        / model / 'LamaH_daily' / f'{station}.csv')
        else:
            return None
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, nrows=1)
            vals = df.iloc[0, :horizon].values.astype(float)
            return pd.Series(vals, name='forecast')
        except Exception:
            return None

    def _simple_forecast(self, ts_monthly: pd.Series, horizon: int) -> pd.Series:
        """Flat mean of last 12 months as naive fallback."""
        baseline_val = float(ts_monthly.tail(12).mean())
        return pd.Series([baseline_val] * horizon, name='forecast')

    def _monthly_relative_anomaly(self, ts: pd.Series) -> pd.Series:
        """Relative anomaly versus monthly climatology."""
        ts = ts.dropna()
        if ts.empty:
            return pd.Series(dtype=float)
        df = ts.to_frame('value')
        df['month'] = df.index.month
        clim = df.groupby('month')['value'].mean()
        denom = df['month'].map(clim).astype(float)
        anomaly = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        safe = denom.abs() > 1e-10
        anomaly.loc[safe] = (df.loc[safe, 'value'] - denom.loc[safe]) / denom.loc[safe].abs()
        return anomaly.rename('anomaly')

    def _compute_sensitivity(self, target_monthly: pd.Series, driver_monthly: pd.Series, direct: bool = False) -> Dict[str, Any]:
        """
        Fit a distributed-lag anomaly response model.
        """
        if direct:
            return {
                'beta': 1.0,
                'elasticity': 1.0,
                'r_value': 1.0,
                'p_value': 0.0,
                'n_months': 0,
                'direct': True,
                'model_type': 'direct_scaling',
                'driver_lag_coeffs': [1.0],
                'target_persistence': 0.0,
                'dominant_lag': 0,
                'cumulative_response': 1.0,
                'residual_std': 0.0,
                'fit_r2': 1.0,
                'used_fallback': False,
            }

        aligned = pd.concat([target_monthly.rename('target'), driver_monthly.rename('driver')], axis=1).dropna()
        if len(aligned) < 24:
            return {
                'beta': 0.0,
                'elasticity': 0.0,
                'r_value': 0.0,
                'p_value': 1.0,
                'n_months': len(aligned),
                'direct': False,
                'model_type': 'insufficient_data',
                'driver_lag_coeffs': [0.0] * (self._MAX_DRIVER_LAGS + 1),
                'target_persistence': 0.0,
                'dominant_lag': 0,
                'cumulative_response': 0.0,
                'residual_std': 0.0,
                'fit_r2': 0.0,
                'used_fallback': True,
            }

        target_anom = self._monthly_relative_anomaly(aligned['target'])
        driver_anom = self._monthly_relative_anomaly(aligned['driver'])
        model_df = pd.DataFrame({'target_anom': target_anom, 'driver_anom': driver_anom}).dropna()
        model_df['target_lag1'] = model_df['target_anom'].shift(1)
        for lag in range(self._MAX_DRIVER_LAGS + 1):
            model_df[f'driver_lag_{lag}'] = model_df['driver_anom'].shift(lag)
        model_df = model_df.dropna()

        if len(model_df) < 18:
            return {
                'beta': 0.0,
                'elasticity': 0.0,
                'r_value': 0.0,
                'p_value': 1.0,
                'n_months': len(model_df),
                'direct': False,
                'model_type': 'insufficient_data',
                'driver_lag_coeffs': [0.0] * (self._MAX_DRIVER_LAGS + 1),
                'target_persistence': 0.0,
                'dominant_lag': 0,
                'cumulative_response': 0.0,
                'residual_std': 0.0,
                'fit_r2': 0.0,
                'used_fallback': True,
            }

        feature_cols = ['target_lag1'] + [f'driver_lag_{lag}' for lag in range(self._MAX_DRIVER_LAGS + 1)]
        X = model_df[feature_cols].to_numpy(dtype=float)
        y = model_df['target_anom'].to_numpy(dtype=float)
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        phi = float(coefs[0])
        driver_lag_coeffs = [float(v) for v in coefs[1:]]
        fitted = X @ coefs
        residuals = y - fitted
        residual_std = float(np.nanstd(residuals)) if len(residuals) else 0.0
        fit_r2 = float(1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) and np.sum((y - y.mean()) ** 2) > 0 else 0.0
        fit_corr = float(np.corrcoef(y, fitted)[0, 1]) if len(y) > 1 else 0.0
        if np.isnan(fit_corr):
            fit_corr = 0.0
        cumulative_response = float(np.sum(driver_lag_coeffs))
        dominant_lag = int(np.argmax(np.abs(driver_lag_coeffs))) if driver_lag_coeffs else 0
        mean_driver = float(aligned['driver'].mean())
        mean_target = float(aligned['target'].mean())
        # Marginal response ratio (linear sensitivity — labelled "elasticity" for legacy compat)
        # Interpretation: % change in target per % change in driver (cumulative, linear).
        marginal_response_ratio = cumulative_response * mean_driver / mean_target if mean_target != 0 else cumulative_response

        # ── F-test p-value for model significance ────────────────────────────
        n_obs = len(y)
        k_params = len(coefs)  # number of regressors
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0 and n_obs > k_params:
            f_stat = ((ss_tot - ss_res) / k_params) / (ss_res / (n_obs - k_params))
            from scipy.stats import f as f_dist
            p_value = float(1.0 - f_dist.cdf(f_stat, k_params, n_obs - k_params))
        else:
            f_stat = 0.0
            p_value = 1.0

        # ── Durbin-Watson statistic ────────────────────────────────────────────
        # DW < 1.5 → positive autocorrelation; 1.5–2.5 → acceptable; > 2.5 → negative
        if len(residuals) > 1:
            dw = float(np.sum(np.diff(residuals) ** 2) / (np.sum(residuals ** 2) + 1e-12))
        else:
            dw = 2.0
        autocorrelation_warning = dw < 1.5 or dw > 2.5

        return {
            'beta': float(cumulative_response),
            'elasticity': float(marginal_response_ratio),           # legacy key
            'marginal_response_ratio': float(marginal_response_ratio),  # descriptive key
            'r_value': fit_corr,
            'p_value': round(p_value, 4),
            'f_stat': round(f_stat, 4),
            'significance': (
                'significant' if p_value < 0.05 else
                'marginal' if p_value < 0.10 else
                'not_significant'
            ),
            'durbin_watson': round(dw, 3),
            'autocorrelation_warning': autocorrelation_warning,
            'n_months': len(model_df),
            'direct': False,
            'model_type': 'distributed_lag_anomaly_response',
            'driver_lag_coeffs': driver_lag_coeffs,
            'target_persistence': phi,
            'dominant_lag': dominant_lag,
            'cumulative_response': cumulative_response,
            'residual_std': residual_std,
            'fit_r2': fit_r2,
            'used_fallback': abs(fit_corr) < 0.2,
            'method_note': (
                'Distributed-lag linear regression with AR(1) persistence (OLS). '
                'Significance tested via F-test; Durbin-Watson statistic checks residual autocorrelation. '
                'Marginal response ratio is a linear sensitivity estimate, not a structural elasticity parameter.'
            ),
        }

    def _simulate_scenario_response(
        self,
        baseline: pd.Series,
        scale_pct: float,
        duration_months: int,
        start_offset: int,
        sensitivity: Dict[str, Any],
        non_negative: bool = True,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        n = len(baseline)
        driver_shock = np.zeros(n, dtype=float)
        window_end = min(start_offset + duration_months, n)
        driver_shock[start_offset:window_end] = scale_pct / 100.0

        response = np.zeros(n, dtype=float)
        if sensitivity.get('direct'):
            response[start_offset:window_end] = scale_pct / 100.0
        else:
            phi = float(sensitivity.get('target_persistence', 0.0))
            lag_coefs = list(sensitivity.get('driver_lag_coeffs', []))
            if not lag_coefs:
                lag_coefs = [0.0] * (self._MAX_DRIVER_LAGS + 1)
            for t in range(n):
                propagated = (phi * response[t - 1]) if t > 0 else 0.0
                for lag, coef in enumerate(lag_coefs):
                    if t - lag >= 0:
                        propagated += float(coef) * driver_shock[t - lag]
                response[t] = np.clip(propagated, -0.95, 3.0)

        floor = 0.0 if non_negative else None
        # Apply the same physical floor to both series so delta is consistent.
        # Without this, a baseline that dips below zero produces a spurious
        # positive delta even when scale_pct == 0.
        baseline_floored = baseline.clip(lower=floor).rename('baseline_floored')
        scenario = (baseline * (1.0 + response)).clip(lower=floor).rename('scenario')
        delta = (scenario - baseline_floored).rename('delta')
        delta_pct = pd.Series(response * 100.0, index=baseline.index, name='delta_pct')
        return scenario, delta, delta_pct, baseline_floored

    def _relationship_strong_enough(self, sensitivity: Dict[str, Any]) -> bool:
        """
        Gate on statistical significance of the distributed-lag model.
        Primary criterion: F-test p-value < 0.10 (marginal significance).
        Secondary: minimum data length and non-trivial cumulative response.
        The heuristic |r|/R² thresholds are retained as secondary guards but
        the F-test p-value is the academically defensible primary criterion.
        """
        if sensitivity.get('direct'):
            return True
        if sensitivity.get('model_type') != 'distributed_lag_anomaly_response':
            return False
        if sensitivity.get('n_months', 0) < 24:
            return False
        # Primary: F-test p-value (statistically defensible)
        p_value = float(sensitivity.get('p_value', 1.0))
        if p_value >= 0.10:
            return False
        # Secondary: cumulative response must be non-trivial
        if abs(float(sensitivity.get('cumulative_response', 0.0))) < 0.03:
            return False
        return True

    # ── main entry point ─────────────────────────────────────────────────────

    def run_scenario(
        self,
        station: str,
        target_feature: str,
        driver_feature: str,
        scale_pct: float,          # e.g. 20 = +20%, -30 = −30%
        duration_months: int,      # how many forecast months the scenario covers
        start_offset: int,         # 0 = first forecast month, 1 = second, etc.
        model: str = 'FlowNet',
        horizon: int = 12,
        include_analysis: bool = False,
    ) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")

        meta = repo.station_index[station]
        features = meta.get('features', [])
        if target_feature not in features:
            raise ValueError(f"'{target_feature}' not available for {station}.")

        unit = repo.feature_units.get(target_feature, '')
        driver_label = driver_feature if driver_feature in features else target_feature

        # ── 1. Load historical target series ─────────────────────────────────
        target_ts = self._load_series(repo, station, target_feature)
        target_monthly = target_ts.resample('MS').mean().dropna()

        # ── 2. Compute sensitivity ────────────────────────────────────────────
        if driver_feature != target_feature:
            if driver_feature not in features:
                raise ValueError(f"Driver feature '{driver_feature}' is not available for station {station}.")
            driver_ts = self._load_series(repo, station, driver_feature)
            driver_monthly = driver_ts.resample('MS').mean().dropna()
        else:
            # Direct scaling — driver IS the target
            driver_monthly = target_monthly.copy()

        is_direct = (driver_feature == target_feature)
        sensitivity = self._compute_sensitivity(target_monthly, driver_monthly, direct=is_direct)
        if not self._relationship_strong_enough(sensitivity):
            raise ValueError('Relationship too weak for a reliable scenario estimate.')

        # ── 3. Build baseline forecast ────────────────────────────────────────
        effective_horizon = max(horizon, start_offset + duration_months)
        csv_fc = self._load_csv_forecast(repo, station, target_feature, model, effective_horizon)

        last_date = target_ts.index[-1]
        freq = repo.feature_frequency.get(target_feature, 'daily')
        if freq == 'monthly':
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=effective_horizon, freq='MS',
            )
        else:
            # Even for daily series work at monthly resolution for scenario
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=effective_horizon, freq='MS',
            )

        if csv_fc is not None:
            n = min(len(csv_fc), effective_horizon)
            baseline_vals = csv_fc.values[:n].tolist()
            # Pad to effective_horizon with last value if needed
            while len(baseline_vals) < effective_horizon:
                baseline_vals.append(baseline_vals[-1])
        else:
            baseline_vals = self._simple_forecast(target_monthly, effective_horizon).tolist()

        baseline = pd.Series(baseline_vals[:effective_horizon], index=forecast_index[:effective_horizon], name='baseline')

        # ── 4. Apply scenario ─────────────────────────────────────────────────
        window_start = start_offset
        window_end = min(start_offset + duration_months, effective_horizon)
        # non_negative: apply a zero floor to both baseline and scenario so that
        # delta is computed symmetrically. Prevents spurious deltas when the ML
        # baseline forecast dips below zero (physically impossible for discharge).
        non_negative = float(target_monthly.dropna().min()) >= 0
        scenario, delta, delta_pct, baseline_floored = self._simulate_scenario_response(
            baseline, scale_pct, duration_months, start_offset, sensitivity,
            non_negative=non_negative,
        )

        # ── 5. Build figure ───────────────────────────────────────────────────
        figure = self._build_figure(
            target_monthly, baseline_floored, scenario, delta, delta_pct,
            station, target_feature, driver_label, scale_pct,
            duration_months, start_offset, unit, sensitivity,
        )

        # ── 6. Build summary stats ────────────────────────────────────────────
        window_baseline = baseline_floored.iloc[window_start:window_end]
        window_scenario = scenario.iloc[window_start:window_end]
        mean_delta = float((window_scenario - window_baseline).mean())
        max_delta = float((window_scenario - window_baseline).abs().max())
        mean_delta_pct = float(delta_pct.iloc[window_start:window_end].mean()) if not delta_pct.iloc[window_start:window_end].isna().all() else 0.0

        baseline_source = 'trained_model_csv' if csv_fc is not None else 'statistical_mean_fallback'
        result = {
            'station': station,
            'target_feature': target_feature,
            'driver_feature': driver_feature,
            'scale_pct': scale_pct,
            'duration_months': duration_months,
            'start_offset': start_offset,
            'model': model,
            'unit': unit,
            'sensitivity': sensitivity,
            'baseline': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in baseline_floored.items()],
            'scenario': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in scenario.items()],
            'delta_abs': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in delta.items()],
            'delta_pct': [{'date': str(d.date()), 'pct': round(float(p), 2)} for d, p in delta_pct.items()],
            'model_note': (
                ('Driver shocks are propagated through a monthly distributed-lag anomaly response model '
                 f'(F-test p={sensitivity.get("p_value", "N/A")}, DW={sensitivity.get("durbin_watson", "N/A")}). '
                 + (' Historical fit is weak — interpret results as indicative only.' if sensitivity.get('used_fallback') else '')
                 + (' Residual autocorrelation detected (DW outside 1.5–2.5); coefficient SEs may be underestimated.' if sensitivity.get('autocorrelation_warning') else ''))
                if not sensitivity.get('direct')
                else 'Direct scaling is applied because the driver and target are the same variable.'
            ),
            'analysis_strength': 'exploratory_sensitivity',
            'caveat': (
                'This is an exploratory distributed-lag sensitivity model, not a causal attribution framework. '
                'Results describe historical statistical relationships and should not be interpreted as '
                'mechanistic flow predictions. Linearity and stationarity are assumed.'
            ),
            'stats': {
                'mean_delta': round(mean_delta, 3),
                'max_delta': round(max_delta, 3),
                'mean_delta_pct': round(mean_delta_pct, 2),
            },
            'figure': plotly.io.to_json(figure),
            'csv_used': csv_fc is not None,          # kept for backward compatibility
            'baseline_source': baseline_source,
        }

        # Generate analysis if requested — always produces output (AI or local fallback)
        if include_analysis:
            result['analysis'] = _generate_scenario_analysis(result)

        return result

    # ── figure builder ────────────────────────────────────────────────────────

    def _build_figure(
        self,
        history: pd.Series,
        baseline: pd.Series,
        scenario: pd.Series,
        delta: pd.Series,
        delta_pct: pd.Series,
        station: str, target_feat: str, driver_feat: str,
        scale_pct: float, duration_months: int, start_offset: int,
        unit: str, sensitivity: Dict[str, Any],
    ) -> go.Figure:
        sign = '+' if scale_pct >= 0 else ''
        driver_label = driver_feat.replace('_', ' ')
        target_label = target_feat.replace('_', ' ')

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.68, 0.32],
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                f'{target_label} forecast: baseline versus scenario',
                f'Absolute change from baseline ({unit})' if unit else 'Absolute change from baseline',
            ),
        )

        BLUE          = '#38bdf8'
        ORANGE        = '#fb923c'
        GREEN_POS     = '#34d399'
        RED_NEG       = '#f87171'
        SCENARIO_FILL = 'rgba(251,146,60,0.10)'

        # Baseline forecast
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values,
            mode='lines', name='Baseline forecast',
            line=dict(color=BLUE, width=2.2, dash='dash'),
            hovertemplate='%{x|%b %Y} — %{y:.2f} ' + unit + '<extra>Baseline</extra>',
        ), row=1, col=1)

        # Scenario forecast
        fig.add_trace(go.Scatter(
            x=scenario.index, y=scenario.values,
            mode='lines+markers', name=f'Scenario ({sign}{scale_pct:.0f}% {driver_label})',
            line=dict(color=ORANGE, width=2.8),
            marker=dict(size=6, color=ORANGE),
            hovertemplate='%{x|%b %Y} — %{y:.2f} ' + unit + '<extra>Scenario</extra>',
        ), row=1, col=1)

        # Filled area between baseline and scenario
        fig.add_trace(go.Scatter(
            x=pd.concat([scenario, baseline[::-1]]).index.tolist(),
            y=pd.concat([scenario, baseline[::-1]]).values.tolist(),
            fill='toself', fillcolor=SCENARIO_FILL,
            line=dict(width=0), showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

        # Shade scenario window
        window_dates = baseline.index[start_offset: start_offset + duration_months]
        if len(window_dates) > 0:
            _vrect_fill = 'rgba(251,146,60,0.07)'
            fig.add_vrect(x0=window_dates[0], x1=window_dates[-1],
                          fillcolor=_vrect_fill, line_width=0, row=1, col=1)
            fig.add_vrect(x0=window_dates[0], x1=window_dates[-1],
                          fillcolor=_vrect_fill, line_width=0, row=2, col=1)
            fig.add_annotation(
                x=window_dates[0],
                y=1.02,
                yref='paper',
                text='Intervention window',
                showarrow=False,
                xanchor='left',
                font=dict(size=10, color=ORANGE),
                bgcolor='rgba(7,17,31,0.75)',
                bordercolor='rgba(251,146,60,0.30)',
                borderwidth=1,
                borderpad=4,
            )

        # Absolute delta bars
        bar_colors = [GREEN_POS if v >= 0 else RED_NEG for v in delta.values]
        fig.add_trace(go.Bar(
            x=delta.index, y=delta.values,
            name='Scenario impact',
            marker_color=bar_colors,
            hovertemplate='%{x|%b %Y}<br>%{y:+.2f} ' + unit + '<extra>Impact</extra>',
        ), row=2, col=1)

        # Zero line for delta
        fig.add_hline(y=0, line_dash='solid', line_color=GRID, line_width=1, row=2, col=1)

        # Elasticity annotation
        mean_pct = float(delta_pct.iloc[start_offset: start_offset + duration_months].mean()) if len(delta_pct) else 0.0
        elast_txt = (
            f"Applied effect: about {mean_pct:+.1f}% during the intervention window  |  "
            f"dominant lag: {sensitivity.get('dominant_lag', 0)} month(s)  |  "
            f"fit R={sensitivity['r_value']:.2f}  |  "
            f"R²={sensitivity.get('fit_r2', 0.0):.2f}"
            if not sensitivity.get('direct')
            else f'Direct scaling applied: about {mean_pct:+.1f}% during the intervention window'
        )

        _note_ann = method_note_annotation(elast_txt, y=-0.12)
        fig.update_layout(**dark_layout(
            title='',
            height=680,
            margin=MARGIN_SUBPLOT,
            hovermode='x unified',
            show_legend=True,
        ))
        fig.update_layout(legend=legend_h())
        fig.add_annotation(**_note_ann)

        _ax = axis_style()
        fig.update_xaxes(**_ax)
        fig.update_yaxes(**_ax)
        fig.update_yaxes(title_text=unit or target_label, row=1, col=1)
        fig.update_yaxes(title_text=unit or 'Change', row=2, col=1)

        style_subplot_titles(fig)

        return fig
