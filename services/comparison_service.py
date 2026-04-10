"""
comparison_service.py
~~~~~~~~~~~~~~~~~~~~~
Basin-level comparison analytics:
  1. Correlation matrix  — Pearson r between all station pairs for one feature
  2. Anomaly leaderboard — stations ranked by deviation from climatology for a year
  3. Basin-wide summary  — aggregated statistics across all stations
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import markdown
import numpy as np
import pandas as pd

from .analysis_service import _gemini_generate
from .data_loader import DataRepository, MultiDataRepository, SeriesRequest


def _generate_component_analysis(component: str, data: Dict[str, Any], feature: str) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    fallback = _fallback_component_analysis(component, data, feature)
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return fallback
    try:
        prompt = _component_prompt(component, data, feature)
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_component_analysis(component, data, feature)


def _component_prompt(component: str, data: Dict[str, Any], feature: str) -> str:
    feature_label = feature.replace('_', ' ')
    if component == 'correlation':
        return (
            'Act as a professional hydrologist writing a detailed interpretation of a basin Spearman correlation matrix.\n\n'
            'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
            '## Matrix Overview\n'
            'Write 4–5 sentences. State the number of stations in the matrix, the primary correlation method (Spearman rank), '
            'why Spearman was chosen for hydrological discharge (right-skewed data), the general level of coherence across the matrix, '
            'and whether the basin behaves as a spatially unified system or as fragmented sub-regions.\n\n'
            '## Strongest Relationships\n'
            'Exactly 5 bullet points. For each of the top correlated station pairs: cite the correlation value, '
            'interpret the hydrological basis (shared catchment forcing, proximity, regulated reservoir influence), '
            'and note whether the relationship reflects physical routing or common climate forcing.\n\n'
            '## Weakest / Divergent Relationships\n'
            'Exactly 4 bullet points. Identify station pairs with notably low or negative correlations, '
            'explain possible causes (independent sub-basins, opposing regulation, data heterogeneity), '
            'and note what these divergences imply for basin-wide monitoring representativeness.\n\n'
            '## Spatial Interpretation\n'
            'Exactly 4 bullet points: (1) which station has the highest mean cross-station correlation and what that implies; '
            '(2) whether highly correlated stations form geographically contiguous clusters; '
            '(3) what the correlation structure implies about spatial transferability of discharge data; '
            '(4) one specific caveat about interpreting high correlation as causation.\n\n'
            '## Operational Implications\n'
            'Exactly 3 bullet points: (1) which stations could serve as basin-wide proxies for monitoring; '
            '(2) how weakly correlated sub-regions should be treated in risk communication; '
            '(3) how correlation structure should inform hydrological model regionalisation or data-sharing decisions.\n\n'
            'RULES:\n'
            '- Cite specific correlation values throughout.\n'
            '- Replace underscores with spaces.\n'
            '- No intro before the first heading; no conclusion after the last bullet.\n\n'
            f'Feature: {feature_label}\n\n'
            f'{_format_correlation_for_prompt(data)}\n'
        )
    if component == 'leaderboard':
        rows = (data or {}).get('rows', [])
        above = (data or {}).get('above_normal', 0)
        below = (data or {}).get('below_normal', 0)
        total = (data or {}).get('total_stations', 0)
        return (
            'Act as a professional hydrologist writing a detailed interpretation of a basin anomaly leaderboard.\n\n'
            'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
            '## Year Context\n'
            'Write 4–5 sentences. State the analysis year, the total number of ranked stations, '
            'the above-normal versus below-normal split and what that balance implies for the overall basin water budget, '
            'name the single station with the largest absolute anomaly and its percentage departure, '
            'and place the year in climatological context (e.g., likely ENSO phase or monsoon character if detectable).\n\n'
            '## Highest Anomaly Stations\n'
            'Exactly 5 bullet points — ranked by absolute anomaly. For each: station name, anomaly percentage, '
            'annual mean versus climatological mean with units, anomaly level classification (critical/warning/watch/normal), '
            'direction (above/below), and a brief hydrological interpretation of what may have driven the departure.\n\n'
            '## Basin Balance & Spatial Pattern\n'
            'Exactly 4 bullet points: (1) sign balance — above-normal vs below-normal count and what a mixed vs unidirectional basin implies; '
            '(2) whether the highest-anomaly stations cluster in a particular sub-region or are spatially dispersed; '
            '(3) whether the magnitudes suggest a uniform basin-wide shift or localised forcing; '
            '(4) comparison with a typical anomaly year — are these departures historically large or modest?\n\n'
            '## Methodology Note\n'
            'Exactly 3 bullet points: (1) how climatology is computed (calendar-month mean average to remove seasonal bias); '
            '(2) the anomaly threshold classification system (±20%, ±40%, ±70%); '
            '(3) what limitations apply when comparing stations with different record lengths.\n\n'
            '## Operational Implications\n'
            'Exactly 3 bullet points: (1) which tier stations warrant immediate follow-up; '
            '(2) how to communicate a mixed above/below-normal basin to water managers; '
            '(3) how the leaderboard should be used as a triage tool, not a definitive severity assessment.\n\n'
            'RULES:\n'
            '- Cite anomaly percentages and discharge means throughout.\n'
            '- Replace underscores with spaces.\n'
            '- Distinguish above-normal and below-normal conditions clearly.\n'
            '- No intro before the first heading; no conclusion after the last bullet.\n\n'
            f'Feature: {feature_label}\n'
            f'Year: {(data or {}).get("year")}\n'
            f'Above normal: {above}, Below normal: {below}, Total: {total}\n\n'
            f'{_format_leaderboard_for_prompt(rows)}\n'
        )
    return (
        'Act as a professional hydrologist writing a detailed interpretation of basin-wide summary statistics.\n\n'
        'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
        '## Basin Snapshot\n'
        'Write 4–5 sentences. State the dataset, feature, and number of active stations; '
        'characterise the basin mean and whether it is high, low, or typical for this type of catchment; '
        'describe the spatial spread (CV, range) and what it implies about within-basin heterogeneity; '
        'mention the data quality context (imputation rate); and state one key operational message from the statistics.\n\n'
        '## Distribution Structure\n'
        'Exactly 5 bullet points: (1) mean and median with units, noting any skew; '
        '(2) standard deviation and spatial CV, interpreting whether the basin is homogeneous or heterogeneous; '
        '(3) full range (min to max) and what the extremes imply about basin diversity; '
        '(4) percentile spread (P10–P90) and what the interquartile width implies for monitoring design; '
        '(5) data quality — average imputation rate and total observations.\n\n'
        '## Station Extremes\n'
        'Exactly 4 bullet points: (1) name and discharge of the highest station, interpret its basin role; '
        '(2) name and discharge of the lowest station, interpret its role; '
        '(3) trend mix (rising/stable/falling counts) and what the dominant trend implies for long-term water availability; '
        '(4) what the ratio of highest to lowest station mean implies about spatial variability in the basin.\n\n'
        '## Interpretive Context\n'
        'Exactly 3 bullet points: (1) how basin mean versus median difference reveals distributional skew; '
        '(2) whether a high spatial CV indicates the summary mean is a reliable representative statistic; '
        '(3) what the trend mix suggests about climate or land-use change signals across the basin.\n\n'
        '## Operational Implications\n'
        'Exactly 3 bullet points: (1) how the distribution spread should inform whether a single basin average is meaningful; '
        '(2) how percentile bands can serve as baseline thresholds for flagging anomalous stations; '
        '(3) how imputation rates should factor into confidence when using these statistics for planning decisions.\n\n'
        'RULES:\n'
        '- Cite all reported statistics with units.\n'
        '- Replace underscores with spaces.\n'
        '- No intro before the first heading; no conclusion after the last bullet.\n\n'
        f'Feature: {feature_label}\n\n'
        f'{_format_summary_for_prompt(data)}\n'
    )


def _format_correlation_for_prompt(corr: Dict[str, Any]) -> str:
    if not corr:
        return 'No correlation matrix available.'
    stations = corr.get('stations', [])
    matrix = corr.get('matrix', [])
    mean_corrs = corr.get('mean_correlations', [])
    pairs: List[tuple[float, str, str]] = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            v = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
            if v is not None:
                pairs.append((float(v), stations[i], stations[j]))
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    top_pairs = pairs[:5]
    top_station = None
    if stations and mean_corrs:
        ranked = [(mean_corrs[i], stations[i]) for i in range(min(len(stations), len(mean_corrs))) if mean_corrs[i] is not None]
        ranked.sort(reverse=True)
        if ranked:
            top_station = ranked[0]
    lines = [
        f"Dataset {corr.get('dataset')} with {corr.get('n_stations')} stations for {corr.get('feature', '')}.",
        f"Capped subset applied: {'yes' if corr.get('capped') else 'no'}; total available stations {corr.get('total_available')}.",
    ]
    if top_station:
        lines.append(f"Highest mean cross-station correlation: {top_station[1]} at {top_station[0]:.3f}.")
    if top_pairs:
        lines.append('Top correlation pairs:')
        lines.extend(f'  - {a} vs {b}: r={v:.3f}' for v, a, b in top_pairs)
    return '\n'.join(lines)


def _format_leaderboard_for_prompt(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return 'No anomaly leaderboard available.'
    lines = ['Top anomaly stations:']
    for row in rows[:8]:
        lines.append(
            f"  - {row['name']}: {row['anomaly_pct']:+.1f}% ({row['direction']} normal), "
            f"year mean {row['year_mean']} {row['unit']} vs climatology {row['clim_mean']} {row['unit']}, level {row['level']}"
        )
    return '\n'.join(lines)


def _format_summary_for_prompt(summary: Dict[str, Any]) -> str:
    if not summary:
        return 'No basin summary available.'
    trends = summary.get('trends', {})
    highest = summary.get('highest_station', {})
    lowest = summary.get('lowest_station', {})
    return (
        f"Dataset: {summary.get('dataset')}\n"
        f"Feature: {summary.get('feature')}\n"
        f"Active stations: {summary.get('active_stations')} of {summary.get('total_stations')}\n"
        f"Basin mean/median/std: {summary.get('basin_mean')} / {summary.get('basin_median')} / {summary.get('basin_std')} {summary.get('unit')}\n"
        f"Range: {summary.get('basin_min')} to {summary.get('basin_max')} {summary.get('unit')}\n"
        f"Percentiles: P10 {summary.get('p10')}, P25 {summary.get('p25')}, P75 {summary.get('p75')}, P90 {summary.get('p90')} {summary.get('unit')}\n"
        f"Spatial CV: {summary.get('spatial_cv_pct')}%\n"
        f"Highest station: {highest.get('name')} ({highest.get('mean')} {summary.get('unit')})\n"
        f"Lowest station: {lowest.get('name')} ({lowest.get('mean')} {summary.get('unit')})\n"
        f"Trend counts: rising {trends.get('rising', 0)}, stable {trends.get('stable', 0)}, falling {trends.get('falling', 0)}\n"
        f"Average imputation: {summary.get('avg_imputation_pct')}%\n"
        f"Total observations: {summary.get('total_observations')}"
    )


def _fallback_component_analysis(component: str, data: Dict[str, Any], feature: str, note: str | None = None) -> str:
    parts: List[str] = []
    feature_label = feature.replace('_', ' ')

    # ── Correlation ──────────────────────────────────────────────────────────
    if component == 'correlation':
        corr = data or {}
        stations = corr.get('stations', [])
        matrix = corr.get('matrix', [])
        mean_corrs = corr.get('mean_correlations', [])
        n_stations = corr.get('n_stations', 0)
        dataset = str(corr.get('dataset', '')).replace('_', ' ')

        pairs: List[tuple] = []
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                v = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
                if v is not None:
                    pairs.append((float(v), stations[i], stations[j]))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        ranked_mean = [(mean_corrs[i], stations[i]) for i in range(min(len(stations), len(mean_corrs))) if mean_corrs[i] is not None]
        ranked_mean.sort(reverse=True)

        all_vals = [v for v, _, _ in pairs]
        mean_all = float(np.mean(all_vals)) if all_vals else 0.0
        coherence = 'strong' if mean_all >= 0.7 else 'moderate' if mean_all >= 0.4 else 'weak'

        parts.append('## Matrix Overview')
        overview = (
            f"The Spearman correlation matrix for **{feature_label}** across the **{dataset}** dataset "
            f"includes {n_stations} station(s). "
            "Spearman rank correlation is used as the primary metric because it is robust to the right-skewness typical of hydrological discharge distributions. "
            f"The overall basin coherence is **{coherence}** (mean pairwise correlation {mean_all:.3f}), "
            f"{'suggesting the basin behaves as a spatially unified system with shared hydroclimatic forcing' if coherence == 'strong' else 'indicating that sub-basin independence or local regulation creates meaningful spatial differentiation across the network'}. "
            "A minimum overlap of 12 months is enforced for all pairs to guard against spurious short-record correlations."
        )
        if note:
            overview += ' ' + note
        parts.append(overview)

        parts.append('## Strongest Relationships')
        for value, left, right in pairs_sorted[:5]:
            strength = 'strong' if abs(value) >= 0.7 else 'moderate' if abs(value) >= 0.4 else 'weak'
            parts.append(
                f'- **{left} vs {right}:** Spearman ρ = {value:.3f} ({strength} coherence) — '
                f'{"likely reflects direct upstream-downstream routing or shared sub-catchment forcing" if abs(value) >= 0.7 else "suggests partial hydrological connectivity, possibly mediated by regulation or differing catchment responses"}.'
            )

        parts.append('## Weakest / Divergent Relationships')
        weakest = sorted(pairs, key=lambda x: abs(x[0]))[:4]
        for value, left, right in weakest:
            parts.append(
                f'- **{left} vs {right}:** ρ = {value:.3f} — '
                f'{"near-zero or negative correlation indicating independent discharge dynamics, possibly driven by opposing regulation, different tributary systems, or contrasting land-use regimes" if abs(value) < 0.3 else "low coherence suggesting different sub-basin characteristics or limited shared forcing"}.'
            )

        parts.append('## Spatial Interpretation')
        if ranked_mean:
            parts.append(
                f'- **Most representative station:** {ranked_mean[0][1]} has the highest mean cross-station correlation at {ranked_mean[0][0]:.3f}, '
                f'making it the best candidate for basin-wide proxy monitoring when full network coverage is unavailable.'
            )
        parts.append(
            f'- **Coverage note:** {"capped subset applied" if corr.get("capped") else "all available stations included"}; '
            f'{corr.get("total_available")} stations are available in total, {n_stations} used in this matrix.'
        )
        parts.append(
            '- **Coherence interpretation:** high pairwise correlations across geographically close stations reflect shared precipitation events and hydrological routing; '
            'weaker correlations across distant or topographically separated stations reflect independent catchment responses.'
        )
        parts.append(
            '- **Causation caveat:** high correlation between two stations does not imply direct hydraulic connection; '
            'it may reflect synchronised climate forcing or parallel basin responses driven by the same meteorological system.'
        )

        parts.append('## Operational Implications')
        parts.append(
            '- **Proxy monitoring:** the most coherent station clusters are suitable candidates for basin-wide discharge monitoring; '
            'a single highly-connected station can represent the network when data transmission failures occur at other sites.'
        )
        parts.append(
            '- **Risk communication:** weakly correlated sub-regions should not be represented by the basin average; '
            'separate sub-basin narratives are needed when communicating conditions during flood or drought events.'
        )
        parts.append(
            '- **Regionalisation:** the correlation structure provides empirical evidence for spatial transferability assumptions in hydrological model calibration and parameter regionalisation exercises.'
        )
        return markdown.markdown('\n'.join(parts))

    # ── Leaderboard ──────────────────────────────────────────────────────────
    if component == 'leaderboard':
        rows = (data or {}).get('rows', [])
        above = (data or {}).get('above_normal', 0)
        below = (data or {}).get('below_normal', 0)
        total = (data or {}).get('total_stations', 0)
        year = (data or {}).get('year', 'the selected year')
        balance = 'above-normal biased' if above > below * 1.5 else 'below-normal biased' if below > above * 1.5 else 'mixed'

        parts.append('## Year Context')
        context = (
            f"The anomaly leaderboard for **{feature_label}** in **{year}** ranks {total} station(s) by their absolute departure from climatological means. "
            f"{above} station(s) are above normal and {below} are below normal, indicating a **{balance}** year across the basin. "
        )
        if rows:
            top = rows[0]
            context += (
                f"The largest single-station departure is **{top['name']}** at {top['anomaly_pct']:+.1f}% relative to climatology, "
                f"classified as **{top['level']}**. "
            )
        context += (
            "Anomaly classification uses calendar-month climatological means to remove the seasonal cycle, "
            "ensuring that departures reflect genuine inter-annual variability rather than seasonal timing shifts."
        )
        if note:
            context += ' ' + note
        parts.append(context)

        parts.append('## Highest Anomaly Stations')
        for row in rows[:5]:
            parts.append(
                f"- **{row['name']}:** {row['anomaly_pct']:+.1f}% ({row['direction']} normal), "
                f"annual mean {row['year_mean']} {row['unit']} versus climatology {row['clim_mean']} {row['unit']}, "
                f"classified as **{row['level']}** — "
                f"{'an extreme departure requiring urgent follow-up analysis' if row['level'] == 'critical' else 'a notable anomaly warranting monitoring attention' if row['level'] == 'warning' else 'an elevated departure worth tracking through the season'}."
            )

        parts.append('## Basin Balance & Spatial Pattern')
        parts.append(
            f"- **Sign balance:** {above} above-normal and {below} below-normal station(s) out of {total} ranked — "
            f"{'a predominantly positive anomaly year suggests basin-wide above-normal water availability' if above > below else 'a predominantly negative anomaly year suggests widespread moisture deficit conditions' if below > above else 'a mixed sign pattern indicates spatially heterogeneous hydroclimatic forcing, not a uniform basin-wide shift'}."
        )
        if rows:
            parts.append(
                f"- **Largest absolute departure:** {rows[0]['name']} at {abs(rows[0]['anomaly_pct']):.1f}% is the basin-wide outlier for {year}; "
                "this station should be prioritised for hydrograph review and cross-referencing with local precipitation or regulation records."
            )
        parts.append(
            "- **Magnitude context:** anomaly classifications (±20% watch, ±40% warning, ±70% critical) are operational heuristics calibrated for hydrological monitoring; "
            "stations in the critical tier indicate departures historically associated with significant water supply or flood risk impacts."
        )
        parts.append(
            "- **Spatial interpretation:** if high-anomaly stations cluster within one sub-region, localised forcing (e.g., ENSO-driven precipitation anomalies, upstream regulation changes) "
            "is more likely than basin-wide climate variability."
        )

        parts.append('## Methodology Note')
        parts.append(
            "- **Climatology computation:** the baseline is derived from calendar-month mean averages across all years of record for each station, "
            "which removes the seasonal cycle and ensures anomalies reflect genuine inter-annual variation."
        )
        parts.append(
            "- **Classification thresholds:** ±20% = watch, ±40% = warning, ±70% = critical — these are operational monitoring heuristics, not statistically derived quantiles, "
            "and do not account for differences in natural flow variability between stations."
        )
        parts.append(
            "- **Record-length sensitivity:** stations with shorter records may have climatological means that are less stable, "
            "leading to anomaly estimates that are more sensitive to individual wet or dry years in the training period."
        )

        parts.append('## Operational Implications')
        parts.append(
            "- **Triage use:** critical and warning-tier stations warrant immediate hydrograph review, cross-checking against local rainfall, and potential escalation to water management authorities."
        )
        parts.append(
            "- **Communication approach:** separate above-normal and below-normal clusters when briefing water managers — "
            "presenting a mixed-sign basin as uniformly 'wet' or 'dry' misrepresents the spatial complexity of conditions."
        )
        parts.append(
            "- **Limitation reminder:** the leaderboard ranks departure magnitude, not operational impact severity; "
            "a 50% anomaly at a small headwater station may be less consequential than a 25% anomaly at a major storage or delta station."
        )
        return markdown.markdown('\n'.join(parts))

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = data or {}
    unit = summary.get('unit', '')
    dataset = str(summary.get('dataset', '')).replace('_', ' ')
    cv = summary.get('spatial_cv_pct', 0)
    mean_val = summary.get('basin_mean')
    median_val = summary.get('basin_median')
    imp_pct = summary.get('avg_imputation_pct', 0)
    trends = summary.get('trends', {})
    highest = summary.get('highest_station', {})
    lowest = summary.get('lowest_station', {})

    parts.append('## Basin Snapshot')
    overview = (
        f"Basin-wide summary statistics for **{feature_label}** across the **{dataset}** dataset "
        f"cover {summary.get('active_stations', 0)} active station(s) out of {summary.get('total_stations', 0)} total. "
        f"The spatial basin mean is {mean_val} {unit} with a median of {median_val} {unit}, "
        f"{'indicating a positively skewed distribution with some high-flow stations pulling the mean above the median' if mean_val and median_val and float(mean_val) > float(median_val) else 'suggesting a relatively symmetric or negatively skewed spatial distribution'}. "
        f"The spatial coefficient of variation is {cv}%, which represents "
        f"{'high spatial heterogeneity — the basin mean is a poor representative statistic and sub-regional analysis is recommended' if float(cv or 0) > 80 else 'moderate spatial heterogeneity — the basin average provides a useful but imprecise summary' if float(cv or 0) > 40 else 'low spatial heterogeneity — the basin mean is a reasonably stable summary for network-wide monitoring'}."
    )
    if note:
        overview += ' ' + note
    parts.append(overview)

    parts.append('## Distribution Structure')
    parts.append(
        f"- **Central tendency:** mean {mean_val} {unit}, median {median_val} {unit} — "
        f"{'mean > median implies right-skewed distribution driven by a few very high-discharge stations' if mean_val and median_val and float(mean_val) > float(median_val) else 'mean ≈ median indicates an approximately symmetric spatial distribution'}."
    )
    parts.append(
        f"- **Spread:** standard deviation {summary.get('basin_std')} {unit} across a full range of "
        f"{summary.get('basin_min')} to {summary.get('basin_max')} {unit}; spatial CV of {cv}%."
    )
    parts.append(
        f"- **Percentile bands:** P10 {summary.get('p10')}, P25 {summary.get('p25')}, P75 {summary.get('p75')}, P90 {summary.get('p90')} {unit} — "
        f"the P10–P90 interquantile range spans {round(float(summary.get('p90') or 0) - float(summary.get('p10') or 0), 3)} {unit}, "
        f"which characterises the typical within-basin station-to-station variability."
    )
    parts.append(
        f"- **Data volume:** {summary.get('total_observations')} total observations across all stations."
    )
    parts.append(
        f"- **Data quality:** average imputation rate is {imp_pct}% — "
        f"{'low imputation suggests high raw data integrity' if float(imp_pct or 0) < 5 else 'moderate imputation; verify imputed segments do not bias trend or extremes analysis' if float(imp_pct or 0) < 20 else 'high imputation rate; interpret summary statistics with caution as imputed values may smooth genuine variability'}."
    )

    parts.append('## Station Extremes')
    if highest:
        parts.append(
            f"- **Highest station:** {highest.get('name')} with mean {highest.get('mean')} {unit} — "
            "likely a high-elevation, high-precipitation catchment or a large mainstem station integrating significant upstream drainage area."
        )
    if lowest:
        parts.append(
            f"- **Lowest station:** {lowest.get('name')} with mean {lowest.get('mean')} {unit} — "
            "likely a small headwater, arid sub-catchment, or tributary with limited drainage area."
        )
    if summary.get('trends_computed') and trends:
        dominant = max(trends, key=trends.get)
        parts.append(
            f"- **Trend mix:** {trends.get('rising', 0)} rising, {trends.get('stable', 0)} stable, {trends.get('falling', 0)} falling — "
            f"a dominant **{dominant}** trend across the network suggests "
            f"{'possible increasing discharge driven by climate change, land-use intensification, or glacial melt contributions' if dominant == 'rising' else 'possible streamflow decline linked to increased evapotranspiration, reduced precipitation, or upstream water abstraction' if dominant == 'falling' else 'general stationarity in the observed period'}."
        )
    if highest and lowest and highest.get('mean') and lowest.get('mean'):
        try:
            ratio = float(highest['mean']) / max(float(lowest['mean']), 1e-9)
            parts.append(
                f"- **Spatial contrast:** the ratio of highest to lowest station mean is {ratio:.1f}×, "
                f"{'indicating extreme within-basin heterogeneity' if ratio > 100 else 'indicating substantial spatial variability' if ratio > 10 else 'indicating moderate spatial variability'}."
            )
        except Exception:
            pass

    parts.append('## Operational Implications')
    parts.append(
        "- **Network planning:** a high spatial CV indicates that a single basin-average statistic is insufficient for operational decisions; "
        "sub-regional groupings or individual station-level analysis should complement basin-wide summaries."
    )
    parts.append(
        "- **Baseline benchmarking:** the P10–P90 percentile band provides a data-driven basis for flagging anomalously high or low station means "
        "in follow-up diagnostics or real-time monitoring dashboards."
    )
    parts.append(
        "- **Decision confidence:** combine summary statistics with station-level anomaly leaderboard and correlation matrix findings before making operational statements about basin-wide conditions."
    )
    return markdown.markdown('\n'.join(parts))


class ComparisonService:
    # Max stations included in the correlation matrix (N² cost)
    CORR_CAP = {'mekong': 65, 'lamah': 50}
    # Skip per-station trend for large datasets to keep summary fast
    TREND_CAP = 200

    def __init__(self, repository: MultiDataRepository) -> None:
        self.repository = repository

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _repo_for(self, dataset: str) -> DataRepository:
        for repo in self.repository.repos:
            if repo.dataset == dataset:
                return repo
        raise ValueError(f'Unknown dataset: {dataset!r}')

    def _stations_for_feature(
        self, dataset: str, feature: str
    ) -> List[Tuple[str, int, Dict]]:
        """
        Return [(station_id, observations, meta)] sorted by observations desc.
        Uses pre-built index — no CSV reads.
        """
        repo = self._repo_for(dataset)
        result = []
        for name, meta in repo.station_index.items():
            fd = meta['feature_details'].get(feature)
            if fd:
                result.append((name, int(fd['observations']), meta))
        return sorted(result, key=lambda x: -x[1])

    def _monthly_series(self, repo: DataRepository, station: str, feature: str) -> pd.Series:
        meta = repo.station_index[station]
        fd = meta['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = repo.get_feature_series(req)
        return (
            df.set_index('Timestamp')['Value']
            .resample('MS').mean()
        )

    def with_component_analysis(self, component: str, data: Dict[str, Any], feature: str) -> Dict[str, Any]:
        enriched = dict(data)
        enriched['analysis'] = _generate_component_analysis(component, data, feature)
        return enriched

    # ── 1. Correlation matrix ────────────────────────────────────────────────

    def compute_correlation_matrix(self, dataset: str, feature: str) -> Dict[str, Any]:
        """
        Pearson and Spearman correlation matrices across all stations (capped for LamaH).
        Resamples every station to monthly means.

        Both metrics are returned:
          - Pearson r: linear association (sensitive to normality assumption).
          - Spearman ρ: monotonic association (robust to skewness — preferred for
            hydrological discharge, which is typically right-skewed).
        The primary reported matrix uses Spearman; Pearson is included for reference.
        min_periods=12 applied to both to guard against spurious short-record correlations.
        """
        from scipy.stats import spearmanr
        repo = self._repo_for(dataset)
        cap = self.CORR_CAP.get(dataset, 65)
        all_info = self._stations_for_feature(dataset, feature)
        total_available = len(all_info)

        if total_available < 2:
            raise ValueError(f'Need at least 2 stations with {feature} data in {dataset}.')

        selected = all_info[:cap]
        unit = repo.feature_units.get(feature, '')

        # Build aligned monthly DataFrame
        frames: Dict[str, pd.Series] = {}
        for name, _, _ in selected:
            try:
                frames[name] = self._monthly_series(repo, name, feature)
            except Exception:
                pass

        if len(frames) < 2:
            raise ValueError('Could not load data for enough stations.')

        combined = pd.DataFrame(frames)
        # Pearson (kept for reference)
        corr_pearson = combined.corr(method='pearson', min_periods=12)
        # Spearman (primary — robust for skewed hydrological data)
        corr_spearman = combined.corr(method='spearman', min_periods=12)

        station_ids = list(corr_spearman.columns)
        pretty = [
            (repo.station_index[s].get('name', s) or s).replace('_', ' ')
            for s in station_ids
        ]

        def _to_matrix(df):
            return [
                [None if pd.isna(v) else round(float(v), 3) for v in row]
                for row in df.reindex(index=station_ids, columns=station_ids).values.tolist()
            ]

        matrix_spearman = _to_matrix(corr_spearman)
        matrix_pearson  = _to_matrix(corr_pearson)
        # Backward-compat: 'matrix' = Spearman (primary)
        matrix = matrix_spearman

        # Mean correlation per station (excluding self) — Spearman
        mean_corrs = []
        n = len(station_ids)
        for i in range(n):
            others = [matrix[i][j] for j in range(n) if i != j and matrix[i][j] is not None]
            mean_corrs.append(round(float(np.mean(others)), 3) if others else None)

        return {
            'stations': pretty,
            'station_ids': station_ids,
            'matrix': matrix,                       # Spearman (primary)
            'matrix_spearman': matrix_spearman,
            'matrix_pearson': matrix_pearson,
            'mean_correlations': mean_corrs,        # Spearman means
            'primary_correlation': 'spearman',
            'correlation_note': (
                'Primary correlation metric is Spearman rank correlation, which is robust to '
                'the right-skewness typical of hydrological discharge series. '
                'Pearson correlation is provided for reference (assumes linearity and normality). '
                'min_periods=12 applied to both.'
            ),
            'feature': feature,
            'unit': unit,
            'dataset': dataset,
            'n_stations': len(station_ids),
            'capped': len(selected) < total_available,
            'total_available': total_available,
        }

    # ── 2. Anomaly leaderboard ───────────────────────────────────────────────

    def compute_anomaly_leaderboard(
        self,
        dataset: str,
        feature: str,
        year: Optional[int] = None,
        top_n: int = 25,
    ) -> Dict[str, Any]:
        """
        Rank all stations by |anomaly %| for the given year vs long-run climatology.
        If year is None, the most recent complete calendar year is used.
        """
        repo = self._repo_for(dataset)
        all_info = self._stations_for_feature(dataset, feature)
        unit = repo.feature_units.get(feature, '')

        rows: List[Dict] = []
        resolved_year: Optional[int] = year

        if resolved_year is None:
            end_date_str = repo.global_time_extent.get('end')
            if end_date_str:
                end_date = pd.to_datetime(end_date_str)
                resolved_year = end_date.year if end_date.month == 12 else end_date.year - 1
            else:
                resolved_year = 2020

        for station_name, _, meta in all_info:
            try:
                monthly = self._monthly_series(repo, station_name, feature).dropna().reset_index()
                monthly.columns = ['Timestamp', 'Value']
                monthly['Year'] = monthly['Timestamp'].dt.year
                monthly['CalMonth'] = monthly['Timestamp'].dt.month

                year_data = monthly[monthly['Year'] == resolved_year]
                if len(year_data) < 6:
                    continue

                year_mean = float(year_data['Value'].mean())
                # Climatology = mean of all calendar-month averages (avoids seasonal bias)
                clim_mean = float(monthly.groupby('CalMonth')['Value'].mean().mean())
                if abs(clim_mean) < 1e-10:
                    continue

                anomaly_pct = (year_mean - clim_mean) / abs(clim_mean) * 100

                abs_pct = abs(anomaly_pct)
                level = (
                    'critical' if abs_pct >= 70 else
                    'warning'  if abs_pct >= 40 else
                    'watch'    if abs_pct >= 20 else
                    'normal'
                )
                display_name = (meta.get('name', station_name) or station_name).replace('_', ' ')

                rows.append({
                    'station': station_name,
                    'name': display_name,
                    'anomaly_pct': round(anomaly_pct, 1),
                    'year_mean': round(year_mean, 3),
                    'clim_mean': round(clim_mean, 3),
                    'level': level,
                    'direction': 'above' if anomaly_pct >= 0 else 'below',
                    'unit': unit,
                })
            except Exception:
                continue

        rows.sort(key=lambda r: abs(r['anomaly_pct']), reverse=True)

        return {
            'year': resolved_year,
            'rows': rows[:top_n],
            'total_stations': len(rows),
            'above_normal': sum(1 for r in rows if r['direction'] == 'above'),
            'below_normal': sum(1 for r in rows if r['direction'] == 'below'),
            'feature': feature,
            'unit': unit,
            'dataset': dataset,
            'threshold_note': (
                'Anomaly classification thresholds (±20%, ±40%, ±70%) are operational '
                'heuristics for hydrological monitoring, not statistically derived bounds. '
                'They indicate the magnitude of departure from climatology relative to '
                'expert-defined severity levels.'
            ),
        }

    # ── 3. Basin-wide summary ────────────────────────────────────────────────

    def compute_basin_summary(self, dataset: str, feature: str) -> Dict[str, Any]:
        """
        Aggregate statistics across all stations using the pre-built index
        (no CSV loads for basic stats) plus optional trend from CSVs.
        """
        repo = self._repo_for(dataset)
        all_info = self._stations_for_feature(dataset, feature)
        unit = repo.feature_units.get(feature, '')

        if not all_info:
            raise ValueError(f'No stations with {feature} data in {dataset}.')

        # ── Basic stats from pre-built index (fast, no CSV) ──────────────────
        station_means: List[Tuple[str, str, float]] = []
        total_obs = 0
        total_imputed = 0

        for name, obs, meta in all_info:
            fd = meta['feature_details'].get(feature, {})
            mean_val = fd.get('mean')
            if mean_val is None:
                continue
            display = (meta.get('name', name) or name).replace('_', ' ')
            station_means.append((name, display, float(mean_val)))
            total_obs += int(fd.get('observations', 0))
            total_imputed += int(fd.get('imputed_points', 0))

        if not station_means:
            raise ValueError('No mean values available for any station.')

        means = [m for _, _, m in station_means]
        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means))
        spatial_cv = round(std_of_means / (abs(mean_of_means) + 1e-12) * 100, 1)

        sorted_by_mean = sorted(station_means, key=lambda x: -x[2])
        highest = sorted_by_mean[0]
        lowest = sorted_by_mean[-1]

        avg_imputation_pct = round(total_imputed / max(total_obs, 1) * 100, 1)

        # ── Trend computation (from CSVs, skipped for very large datasets) ───
        trends = {'rising': 0, 'stable': 0, 'falling': 0}
        compute_trends = len(all_info) <= self.TREND_CAP

        if compute_trends:
            for name, _, meta in all_info:
                try:
                    monthly = self._monthly_series(repo, name, feature).dropna()
                    if len(monthly) < 12:
                        continue
                    x = np.arange(len(monthly))
                    slope = float(np.polyfit(x, monthly.values, 1)[0])
                    rel = slope / (abs(monthly.mean()) + 1e-12)
                    if rel > 0.001:
                        trends['rising'] += 1
                    elif rel < -0.001:
                        trends['falling'] += 1
                    else:
                        trends['stable'] += 1
                except Exception:
                    continue

        # ── Distribution histogram (30 bins) ─────────────────────────────────
        counts, edges = np.histogram(means, bins=min(30, len(means)))

        # ── Percentile bands ─────────────────────────────────────────────────
        p10 = float(np.percentile(means, 10))
        p25 = float(np.percentile(means, 25))
        p75 = float(np.percentile(means, 75))
        p90 = float(np.percentile(means, 90))

        return {
            'dataset': dataset,
            'feature': feature,
            'unit': unit,
            'active_stations': len(station_means),
            'total_stations': len(all_info),
            'basin_mean': round(mean_of_means, 3),
            'basin_median': round(float(np.median(means)), 3),
            'basin_std': round(std_of_means, 3),
            'basin_min': round(float(np.min(means)), 3),
            'basin_max': round(float(np.max(means)), 3),
            'p10': round(p10, 3),
            'p25': round(p25, 3),
            'p75': round(p75, 3),
            'p90': round(p90, 3),
            'spatial_cv_pct': spatial_cv,
            'highest_station': {'id': highest[0], 'name': highest[1], 'mean': round(highest[2], 3)},
            'lowest_station': {'id': lowest[0], 'name': lowest[1], 'mean': round(lowest[2], 3)},
            'trends': trends,
            'trends_computed': compute_trends,
            'avg_imputation_pct': avg_imputation_pct,
            'total_observations': total_obs,
            'histogram': {
                'counts': counts.tolist(),
                'edges': [round(float(e), 3) for e in edges.tolist()],
            },
        }

    # ── Combined entry point ─────────────────────────────────────────────────

    def compare(self, dataset: str, feature: str, year: Optional[int] = None, include_analysis: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'correlation': None,
            'leaderboard': None,
            'summary': None,
            'errors': [],
        }
        for key, fn in [
            ('correlation', lambda: self.compute_correlation_matrix(dataset, feature)),
            ('leaderboard', lambda: self.compute_anomaly_leaderboard(dataset, feature, year)),
            ('summary',     lambda: self.compute_basin_summary(dataset, feature)),
        ]:
            try:
                component_result = fn()
                if include_analysis:
                    component_result = self.with_component_analysis(key, component_result, feature)
                result[key] = component_result
            except Exception as exc:
                result['errors'].append(f'{key}: {exc}')
        return result
