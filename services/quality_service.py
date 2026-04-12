from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots

import markdown

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest


def generate_quality_analysis(view: str, result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    fallback = _fallback_quality_analysis(view, result)
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return fallback
    try:
        prompt = _quality_prompt(view, result)
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_quality_analysis(view, result)


def _quality_prompt(view: str, result: Dict[str, Any]) -> str:
    if view == 'completeness':
        overall = result.get('overall_pct', 0)
        completeness_tier = 'high' if float(overall or 0) >= 90 else 'moderate' if float(overall or 0) >= 70 else 'low'
        return (
            'Act as a professional hydrologist performing a detailed data-quality completeness audit.\n\n'
            'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
            '## Quality Summary\n'
            'Write 4–5 sentences. State the station and feature, the overall completeness percentage and tier (high/moderate/low), '
            'how many months are fully missing versus partially complete, what this completeness level implies for the reliability of seasonal and trend analyses, '
            'and one key recommendation for the data user.\n\n'
            '## Coverage Assessment\n'
            'Exactly 5 bullet points: (1) overall completeness rate with tier label; '
            '(2) number of fully missing months and share of the total record; '
            '(3) number of months below 50% completeness and what that implies; '
            '(4) the total evaluated record length in months; '
            '(5) whether gaps appear concentrated in a particular season or period, and the implication for seasonal analysis.\n\n'
            '## Reliability Implications\n'
            'Exactly 4 bullet points: (1) impact on extreme value estimation; '
            '(2) impact on trend detection (short or gappy records bias regression slopes); '
            '(3) impact on seasonal decomposition and climatological baseline computation; '
            '(4) impact on cross-station correlation analysis (short common overlap periods).\n\n'
            '## Recommended Handling\n'
            'Exactly 3 bullet points: (1) whether to proceed with analysis or flag for data recovery; '
            '(2) specific analysis methods that require caution or are inappropriate at this completeness level; '
            '(3) recommended quality-control steps before using this series in any downstream hydrological modelling.\n\n'
            'RULES:\n'
            '- Cite all provided completeness statistics.\n'
            '- Replace underscores with spaces.\n'
            '- Technical tone throughout. No intro before the first heading.\n\n'
            f"Station: {result.get('station', '').replace('_', ' ')}\n"
            f"Feature: {result.get('feature', '').replace('_', ' ')}\n"
            f"Overall completeness: {overall}% ({completeness_tier})\n"
            f"Missing months: {result.get('missing_months')}\n"
            f"Low-completeness months (<50%): {result.get('low_months')}\n"
            f"Total months evaluated: {result.get('total_months')}\n"
        )
    if view == 'imputation':
        top_rows = (result.get('rows') or [])[:8]
        top_text = '\n'.join(
            f"- {r['name']} · {r['feature'].replace('_', ' ')}: {r['imp_pct']:.1f}% imputed "
            f"({r['imputed']} of {r['observations']})"
            for r in top_rows
        ) or 'No station rows available.'
        imp_pct = result.get('overall_imp_pct', 0)
        imp_tier = 'low' if float(imp_pct or 0) < 5 else 'moderate' if float(imp_pct or 0) < 20 else 'high'
        return (
            'Act as a professional hydrologist performing a detailed imputation audit across a basin dataset.\n\n'
            'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
            '## Quality Summary\n'
            'Write 4–5 sentences. State the dataset and feature filter, the overall imputation rate and tier (low/moderate/high), '
            'how many stations are affected and how many have high imputation (≥20%), '
            'identify the most heavily imputed station and its rate, '
            'and characterise the implications for basin-wide data integrity.\n\n'
            '## Imputation Distribution\n'
            'Exactly 5 bullet points: (1) overall imputation rate and total imputed observations; '
            '(2) number of stations with any imputation and their share of the total; '
            '(3) number of high-imputation stations (≥20%) and the risk they pose; '
            '(4) the top affected station with its imputation percentage; '
            '(5) whether imputation is concentrated in specific stations or spread broadly across the dataset.\n\n'
            '## Reliability Implications\n'
            'Exactly 4 bullet points: (1) how high imputation rates can bias mean and variance estimates; '
            '(2) how imputed segments may mask genuine hydrological extremes or gaps; '
            '(3) how correlation-based analyses can be affected if imputation algorithms propagate smoothed values; '
            '(4) how trend detection is affected when imputation fills long gaps with climatological means.\n\n'
            '## Recommended Handling\n'
            'Exactly 3 bullet points: (1) for low-imputation stations — analysis can proceed with standard methods; '
            '(2) for moderate/high-imputation stations — specific precautions needed; '
            '(3) a QC recommendation for stations flagged as high-imputation before use in operational products.\n\n'
            'RULES:\n'
            '- Cite all provided statistics and station names.\n'
            '- Replace underscores with spaces.\n'
            '- Technical tone. No intro before the first heading.\n\n'
            f"Dataset: {result.get('dataset')}\n"
            f"Feature filter: {(result.get('feature') or 'all features').replace('_', ' ')}\n"
            f"Overall imputation: {imp_pct}% ({imp_tier})\n"
            f"Total observations: {result.get('total_observations')}\n"
            f"Total imputed: {result.get('total_imputed')}\n"
            f"Stations with imputation: {result.get('stations_with_imputation')}\n"
            f"High-imputation stations (>=20%): {result.get('high_imputation_stations')}\n"
            f"Top affected rows:\n{top_text}\n"
        )
    if view == 'gaps':
        gap_rows = (result.get('gaps') or [])[:8]
        gap_text = '\n'.join(
            f"- {g['start']} to {g['end']}: {g['length']} {g['unit']} ({g['severity']})"
            for g in gap_rows
        ) or 'No gaps recorded.'
        missing_pct = result.get('missing_pct', 0)
        gap_tier = 'minor' if float(missing_pct or 0) < 5 else 'moderate' if float(missing_pct or 0) < 20 else 'severe'
        return (
            'Act as a professional hydrologist performing a detailed temporal gap analysis for a discharge station.\n\n'
            'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
            '## Quality Summary\n'
            'Write 4–5 sentences. State the station and feature, the total missing data share and severity tier, '
            'the number and severity breakdown of detected gaps, identify the longest gap and its start-to-end range, '
            'and state the primary impact on hydrological analysis at this site.\n\n'
            '## Gap Structure\n'
            'Exactly 5 bullet points: (1) total missing share with severity tier; '
            '(2) total gap count and severity breakdown (major/moderate/minor); '
            '(3) the longest gap — dates, length, and severity; '
            '(4) the second and third largest gaps, if available; '
            '(5) whether gaps are concentrated in a particular season or time period.\n\n'
            '## Reliability Implications\n'
            'Exactly 4 bullet points: (1) how major gaps affect annual statistics and trend detection; '
            '(2) how gaps affect seasonal index computation (e.g., blocking complete-year requirements); '
            '(3) how gaps in the most recent period affect current-condition monitoring; '
            '(4) how total missing share interacts with imputation — are gaps filled or left as NaN?\n\n'
            '## Recommended Handling\n'
            'Exactly 3 bullet points: (1) whether data recovery should be pursued and from which sources; '
            '(2) what gap-filling methods are appropriate given the severity and duration; '
            '(3) how analysts should flag gap-affected periods in reports and operational outputs.\n\n'
            'RULES:\n'
            '- Cite all gap counts, dates, and durations.\n'
            '- Replace underscores with spaces.\n'
            '- Technical tone. No intro before the first heading.\n\n'
            f"Station: {result.get('station', '').replace('_', ' ')}\n"
            f"Feature: {result.get('feature', '').replace('_', ' ')}\n"
            f"Missing share: {missing_pct}% ({gap_tier})\n"
            f"Gap count: {result.get('gap_count')}\n"
            f"Major gaps: {result.get('major')}, Moderate: {result.get('moderate')}, Minor: {result.get('minor')}\n"
            f"Largest gaps:\n{gap_text}\n"
        )
    # anomalies
    candidates = (result.get('candidates') or [])[:8]
    cand_text = '\n'.join(
        f"- {c['date']}: value {c['value']} {result.get('unit', '')}, |z|={c['z_score']}, "
        f"{'imputed' if c.get('is_imputed') else 'observed'}, flag={c.get('flag')}"
        for c in candidates
    ) or 'No anomaly candidates.'
    total_cands = result.get('total', 0)
    unflagged = result.get('unflagged', 0)
    return (
        'Act as a professional hydrologist performing a z-score anomaly candidate review for a discharge station.\n\n'
        'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
        '## Quality Summary\n'
        'Write 4–5 sentences. State the station, feature, and z-score threshold used for detection; '
        'report the total number of anomaly candidates and how many remain unflagged; '
        'characterise the mean and standard deviation of the series and what the detection threshold implies; '
        'note whether the proportion of flagged to unflagged candidates is high or low; '
        'and state one key operational implication for data use.\n\n'
        '## Candidate Structure\n'
        'Exactly 5 bullet points: (1) total candidate count and detection threshold; '
        '(2) unflagged vs already-flagged split; '
        '(3) the most extreme candidate — date, value, z-score, and observed vs imputed status; '
        '(4) whether candidates cluster in a particular time period; '
        '(5) the proportion of imputed versus observed candidates and what that implies.\n\n'
        '## Reliability Implications\n'
        'Exactly 4 bullet points: (1) how unreviewed candidates affect mean and variance estimates; '
        '(2) how extreme-value analyses (GEV, return periods) are compromised by uninspected outliers; '
        '(3) how imputed anomaly candidates are a double quality concern; '
        '(4) how the current flagging rate affects the trustworthiness of downstream trend and correlation analyses.\n\n'
        '## Recommended Handling\n'
        'Exactly 3 bullet points: (1) how to prioritise review — by z-score magnitude, by imputed status, or by time period; '
        '(2) what evidence should be checked to confirm or dismiss each candidate (cross-station comparison, precipitation records, gauge notes); '
        '(3) how to document anomaly decisions in the quality-flag system for future users.\n\n'
        'RULES:\n'
        '- Cite all candidate counts, z-scores, and flag statuses.\n'
        '- Replace underscores with spaces.\n'
        '- Technical tone. No intro before the first heading.\n\n'
        f"Station: {result.get('station', '').replace('_', ' ')}\n"
        f"Feature: {result.get('feature', '').replace('_', ' ')}\n"
        f"Z threshold: {result.get('z_thresh')}\n"
        f"Series mean: {result.get('mean')}, std: {result.get('std')}\n"
        f"Total candidates: {total_cands}, Unflagged: {unflagged}\n"
        f"Candidate examples:\n{cand_text}\n"
    )


def _fallback_quality_analysis(view: str, result: Dict[str, Any], note: str | None = None) -> str:
    parts: list = []

    # ── Completeness ─────────────────────────────────────────────────────────
    if view == 'completeness':
        station = result.get('station', '').replace('_', ' ')
        feature = result.get('feature', '').replace('_', ' ')
        overall = result.get('overall_pct', 0)
        missing = result.get('missing_months', 0)
        low = result.get('low_months', 0)
        total_months = result.get('total_months', 0)
        completeness_tier = 'high' if float(overall or 0) >= 90 else 'moderate' if float(overall or 0) >= 70 else 'low'

        parts.append('## Quality Summary')
        intro = (
            f"Temporal completeness audit for **{station}** — **{feature}** over {total_months} evaluated monthly periods. "
            f"Overall completeness is **{overall}%**, classified as **{completeness_tier}**. "
            f"{missing} months are fully absent from the record, and {low} additional months fall below 50% daily coverage. "
        )
        if completeness_tier == 'high':
            intro += "The record quality is sufficient for seasonal analysis, trend detection, and extreme-value estimation with standard methods."
        elif completeness_tier == 'moderate':
            intro += "This level of completeness is adequate for broad analysis but requires caution in periods adjacent to data voids."
        else:
            intro += "The low completeness significantly limits analytical reliability; data recovery or gap-filling should be prioritised before applying this series in operational products."
        if note:
            intro += ' ' + note
        parts.append(intro)

        parts.append('## Coverage Assessment')
        parts.append(
            f"- **Overall completeness:** {overall}% ({completeness_tier}) — "
            f"{'sufficient for all standard hydrological analyses' if completeness_tier == 'high' else 'adequate for trend and seasonal analysis with caveats' if completeness_tier == 'moderate' else 'insufficient for reliable extreme-value or trend analysis without supplementary data'}."
        )
        parts.append(
            f"- **Fully missing months:** {missing} of {total_months} evaluated months have no data at all, representing {round(float(missing or 0) / max(float(total_months or 1), 1) * 100, 1)}% of the total record period."
        )
        parts.append(
            f"- **Partially complete months:** {low} months have fewer than 50% of expected daily observations, introducing uncertainty into monthly aggregates derived from those periods."
        )
        parts.append(
            f"- **Total record evaluated:** {total_months} months; a longer record with high completeness provides more stable climatological baselines and more reliable extreme-value parameter estimates."
        )
        parts.append(
            "- **Seasonal distribution:** concentrated gaps in particular seasons (e.g., monsoon peaks or dry-season minima) can systematically bias seasonal statistics; "
            "examine the completeness heatmap for temporal clustering of missing periods before seasonal analysis."
        )

        parts.append('## Reliability Implications')
        parts.append(
            "- **Extreme value estimation:** incomplete records reduce the effective sample size for annual maximum series, potentially underestimating or overestimating return-period flood levels; "
            "GEV/Gumbel fits should be validated against the completeness record before operationalising design-flood estimates."
        )
        parts.append(
            "- **Trend detection:** missing data introduce artificial step changes and bias ordinary least squares trend slopes; "
            "Mann-Kendall tests are more robust to gaps but still require a minimum record length for statistical power."
        )
        parts.append(
            "- **Seasonal decomposition:** STL and calendar-month climatologies require consistent multi-year coverage; "
            "months with repeated gaps will have unreliable seasonal component estimates."
        )
        parts.append(
            "- **Cross-station correlation:** the common overlap period between this station and others may be shorter than the individual records; "
            "ensure min_periods requirements are applied when computing Pearson or Spearman correlations."
        )

        parts.append('## Recommended Handling')
        parts.append(
            f"- **Proceed vs. flag:** {'analysis can proceed with standard methods; note missing periods in any report outputs' if completeness_tier == 'high' else 'proceed with caution; clearly annotate all figures with the missing-period timeline' if completeness_tier == 'moderate' else 'flag this station for data recovery before operational use; limit analysis to sub-periods with ≥80% completeness'}."
        )
        parts.append(
            "- **Incompatible methods:** at completeness levels below 70%, avoid using this series as a standalone input for frequency analysis or trend attribution; "
            "if it must be used, apply bootstrapped confidence intervals and clearly report the effective sample size."
        )
        parts.append(
            "- **QC steps:** cross-check missing periods against gauge event logs, rating curve revision dates, and regional precipitation anomaly records; "
            "determine whether gaps are random (instrument failure) or systematic (seasonal access limitations) before choosing a gap-filling strategy."
        )
        return markdown.markdown('\n'.join(parts))

    # ── Imputation ────────────────────────────────────────────────────────────
    if view == 'imputation':
        dataset = str(result.get('dataset', '')).replace('_', ' ')
        feat_filter = (result.get('feature') or 'all features').replace('_', ' ')
        overall_imp = result.get('overall_imp_pct', 0)
        total_obs = result.get('total_observations', 0)
        total_imp = result.get('total_imputed', 0)
        stations_imp = result.get('stations_with_imputation', 0)
        high_imp = result.get('high_imputation_stations', 0)
        rows = (result.get('rows') or [])
        imp_tier = 'low' if float(overall_imp or 0) < 5 else 'moderate' if float(overall_imp or 0) < 20 else 'high'

        parts.append('## Quality Summary')
        intro = (
            f"Imputation audit for the **{dataset}** dataset (feature: **{feat_filter}**). "
            f"Across {total_obs:,} total observations, {total_imp:,} values ({overall_imp}%) are flagged as imputed — a **{imp_tier}** imputation rate. "
            f"{stations_imp} station(s) contain at least one imputed observation, of which {high_imp} have rates at or above the 20% high-imputation threshold. "
        )
        if rows:
            top = rows[0]
            intro += (
                f"The most heavily imputed station is **{top['name']}** with {top['imp_pct']:.1f}% imputation ({top['imputed']} of {top['observations']} observations). "
            )
        if imp_tier == 'low':
            intro += "Overall data integrity is high; imputed values are unlikely to materially affect basin-wide analysis results."
        elif imp_tier == 'moderate':
            intro += "Moderate imputation requires targeted caution for stations in the high-imputation tier before applying extreme-value or trend analyses."
        else:
            intro += "High basin-wide imputation rates indicate significant reliance on reconstructed data; all downstream analyses should document this limitation explicitly."
        if note:
            intro += ' ' + note
        parts.append(intro)

        parts.append('## Imputation Distribution')
        parts.append(
            f"- **Overall rate:** {overall_imp}% ({imp_tier}) — {total_imp:,} of {total_obs:,} observations are imputed across the dataset."
        )
        parts.append(
            f"- **Station coverage:** {stations_imp} station(s) have at least one imputed observation; "
            f"{high_imp} station(s) are classified as high-imputation (≥20%), posing the greatest analytical risk."
        )
        if rows:
            parts.append(
                f"- **Highest imputation station:** {rows[0]['name']} · {rows[0]['feature'].replace('_', ' ')} — {rows[0]['imp_pct']:.1f}% ({rows[0]['imputed']} of {rows[0]['observations']} observations)."
            )
            if len(rows) > 1:
                parts.append(
                    f"- **Second highest:** {rows[1]['name']} · {rows[1]['feature'].replace('_', ' ')} — {rows[1]['imp_pct']:.1f}% imputation."
                )
        parts.append(
            "- **Distribution pattern:** if high-imputation stations cluster in a particular sub-region or feature type, "
            "it may indicate systemic data-collection issues (e.g., telemetry failures, seasonal inaccessibility, or rating curve uncertainty) rather than isolated instrument faults."
        )

        parts.append('## Reliability Implications')
        parts.append(
            "- **Mean and variance bias:** imputation algorithms (typically climatological infilling or nearby-station interpolation) introduce smoothed values that underestimate true variance; "
            "standard deviations and CV estimates at high-imputation stations should be treated as lower bounds."
        )
        parts.append(
            "- **Extreme value masking:** if imputation fills a genuine flood or drought episode with a climatological mean, the extreme will be absent from the analysis series, "
            "causing return-period estimates to underestimate the true risk."
        )
        parts.append(
            "- **Correlation effects:** correlation-based analyses can be biased if imputation propagates smoothed values that artificially inflate cross-station coherence, "
            "particularly if the same infilling source is used for multiple stations."
        )
        parts.append(
            "- **Trend bias:** imputation with climatological means effectively fixes the imputed periods to historical average levels, "
            "which can weaken genuine long-term trends or create artificial breakpoints at imputation boundaries."
        )

        parts.append('## Recommended Handling')
        parts.append(
            f"- **Low-imputation stations (<5%):** standard analysis methods apply; note imputed segments in report footnotes but no special treatment is required."
        )
        parts.append(
            "- **Moderate-to-high imputation stations (≥20%):** apply sensitivity tests by comparing analysis results with and without imputed periods; "
            "consider excluding high-imputation segments from extreme-value or trend analyses where alternative records are unavailable."
        )
        parts.append(
            "- **Data recovery recommendation:** for the highest-imputation stations, pursue original source data from gauge operators, national hydrology agencies, or satellite-derived discharge products "
            "before incorporating these series in operational planning documents or design-flood reports."
        )
        return markdown.markdown('\n'.join(parts))

    # ── Gaps ─────────────────────────────────────────────────────────────────
    if view == 'gaps':
        station = result.get('station', '').replace('_', ' ')
        feature = result.get('feature', '').replace('_', ' ')
        missing_pct = result.get('missing_pct', 0)
        gap_count = result.get('gap_count', 0)
        major = result.get('major', 0)
        moderate = result.get('moderate', 0)
        minor = result.get('minor', 0)
        gaps = result.get('gaps') or []
        gap_tier = 'minor' if float(missing_pct or 0) < 5 else 'moderate' if float(missing_pct or 0) < 20 else 'severe'

        parts.append('## Quality Summary')
        intro = (
            f"Temporal gap analysis for **{station}** — **{feature}**. "
            f"The record has {missing_pct}% missing data across {gap_count} detected gap(s), classified as a **{gap_tier}** gap structure. "
            f"Severity breakdown: {major} major, {moderate} moderate, and {minor} minor gap(s). "
        )
        if gaps:
            g0 = gaps[0]
            intro += (
                f"The longest gap spans {g0['start']} to {g0['end']} ({g0['length']} {g0['unit']}, {g0['severity']}), "
                "which represents the most significant data void for continuity-sensitive analyses. "
            )
        if gap_tier == 'minor':
            intro += "The overall gap burden is low; most standard analyses can proceed without modification."
        elif gap_tier == 'moderate':
            intro += "Moderate gaps require targeted handling for extremes and trend analyses adjacent to the void periods."
        else:
            intro += "The severe gap burden substantially limits analytical reliability; data recovery or gap-filling is strongly recommended."
        if note:
            intro += ' ' + note
        parts.append(intro)

        parts.append('## Gap Structure')
        parts.append(
            f"- **Total missing share:** {missing_pct}% ({gap_tier}) — across {gap_count} detected discontinuity event(s) in the record."
        )
        parts.append(
            f"- **Severity breakdown:** major {major} (>30-day voids), moderate {moderate} (7–30 days), minor {minor} (<7 days)."
        )
        if gaps:
            g0 = gaps[0]
            parts.append(
                f"- **Largest gap:** {g0['start']} to {g0['end']}, spanning {g0['length']} {g0['unit']} — classified as {g0['severity']}."
            )
            if len(gaps) > 1:
                g1 = gaps[1]
                parts.append(
                    f"- **Second largest gap:** {g1['start']} to {g1['end']}, spanning {g1['length']} {g1['unit']} — classified as {g1['severity']}."
                )
            else:
                parts.append("- **Additional gaps:** only one significant gap detected; the rest of the record is continuous.")
        parts.append(
            "- **Temporal clustering:** examine whether gaps cluster during monsoon or dry seasons — "
            "seasonal clustering can bias monthly and annual statistics even when the total missing share appears modest."
        )

        parts.append('## Reliability Implications')
        parts.append(
            "- **Annual statistics:** major gaps spanning multiple months may exclude complete wet or dry seasons from annual means, "
            "biasing long-term averages and potentially distorting return-period estimates."
        )
        parts.append(
            "- **Trend detection:** gaps create artificial discontinuities in regression-based trend methods; "
            "Mann-Kendall trend tests are more robust, but large central-record gaps can still invalidate the test's stationarity assumption."
        )
        parts.append(
            "- **Current-condition monitoring:** if the most recent gap is within the last 12 months, real-time risk classification and anomaly detection for this station are unreliable until data is restored."
        )
        parts.append(
            "- **Gap versus imputation interaction:** gaps that have been filled by imputation show as complete in the completeness view but may introduce smoothed values into analysis; "
            "cross-reference with the imputation audit to determine how much of the missing period was reconstructed."
        )

        parts.append('## Recommended Handling')
        parts.append(
            "- **Data recovery:** contact the gauge authority or national hydrological agency to recover raw data for major gap periods; "
            "satellite-based precipitation-runoff models (e.g., GloFAS reanalysis) can serve as interim gap-filling references."
        )
        parts.append(
            "- **Gap-filling strategy:** for minor gaps (<7 days), linear interpolation or nearby-station regression is appropriate; "
            "for major gaps, climatological infilling using long-run calendar-month means is preferred, but the filled values should be clearly flagged in the data record."
        )
        parts.append(
            "- **Reporting:** all analysis outputs derived from this station should document the gap timeline; "
            "figures should overlay gap periods as shaded regions to prevent misinterpretation by downstream users."
        )
        return markdown.markdown('\n'.join(parts))

    # ── Anomalies ─────────────────────────────────────────────────────────────
    station = result.get('station', '').replace('_', ' ')
    feature = result.get('feature', '').replace('_', ' ')
    z_thresh = result.get('z_thresh', 3.0)
    mean_val = result.get('mean')
    std_val = result.get('std')
    total_cands = result.get('total', 0)
    unflagged = result.get('unflagged', 0)
    candidates = result.get('candidates') or []
    flagged = int(total_cands or 0) - int(unflagged or 0)
    imp_cands = sum(1 for c in candidates if c.get('is_imputed'))

    parts.append('## Quality Summary')
    intro = (
        f"Z-score anomaly candidate review for **{station}** — **{feature}**. "
        f"Detection threshold: |z| ≥ {z_thresh} (series mean: {mean_val}, std: {std_val}). "
        f"{total_cands} candidate(s) were identified, of which {unflagged} remain unflagged and {flagged} have already been reviewed. "
    )
    if candidates:
        top = candidates[0]
        intro += (
            f"The most extreme candidate is on **{top['date']}** with value {top['value']} and |z| = {top['z_score']}, "
            f"which is {'imputed' if top.get('is_imputed') else 'an observed value'}. "
        )
    if unflagged == 0:
        intro += "All identified candidates have been reviewed — no outstanding anomaly decisions remain."
    elif int(unflagged or 0) > 5:
        intro += "A significant number of candidates remain unreviewed; these should be prioritised before using this series in downstream analyses."
    else:
        intro += "A small number of unflagged candidates remain; targeted review is recommended before finalising quality flags."
    if note:
        intro += ' ' + note
    parts.append(intro)

    parts.append('## Candidate Structure')
    parts.append(
        f"- **Total detected:** {total_cands} candidate(s) at |z| ≥ {z_thresh}; flagged {flagged}, unflagged {unflagged}."
    )
    parts.append(
        f"- **Review status:** {'all candidates reviewed — good quality control coverage' if unflagged == 0 else f'{unflagged} candidate(s) still require review before this station is approved for operational use'}."
    )
    if candidates:
        top = candidates[0]
        parts.append(
            f"- **Most extreme candidate:** {top['date']} — value {top['value']}, |z| = {top['z_score']}, "
            f"{'imputed value (double quality concern: anomalous AND reconstructed)' if top.get('is_imputed') else 'observed value (instrument error or genuine extreme event)'}."
        )
        parts.append(
            f"- **Imputed candidates:** {imp_cands} of the listed candidates are imputed values, "
            "indicating the imputation algorithm generated an anomalous estimate; these require verification against the imputation source data."
        )
    parts.append(
        "- **Temporal clustering:** if candidates cluster near known event dates (e.g., typhoon passages, dam releases, rating-curve revisions), they may reflect real events rather than instrument errors and should be retained after review."
    )

    parts.append('## Reliability Implications')
    parts.append(
        "- **Mean and variance:** unreviewed high-z candidates artificially inflate the standard deviation and may shift the mean; "
        "sensitive analyses (e.g., correlation matrices) should be run both with and without candidate exclusion as a sensitivity check."
    )
    parts.append(
        "- **Extreme value analysis:** uninspected outliers in the annual maximum series will directly affect GEV and Gumbel parameter estimates and return-period flood levels; "
        "QC must be completed before any frequency analysis is performed."
    )
    parts.append(
        "- **Imputed anomaly candidates:** a value that is both imputed and has |z| > threshold indicates the reconstruction algorithm failed for that period; "
        "these points carry compounded uncertainty and should be removed from analysis unless corroborated."
    )
    parts.append(
        "- **Flagging coverage:** a high ratio of unflagged to total candidates indicates incomplete quality control, "
        "which reduces user confidence in the integrity of the full dataset and limits its suitability for operational flood or drought reporting."
    )

    parts.append('## Recommended Handling')
    parts.append(
        "- **Review prioritisation:** start with the highest |z| candidates and all imputed anomalies; "
        "cross-check each against nearby-station discharge records, satellite precipitation estimates, and gauge event logs before deciding to retain, correct, or remove."
    )
    parts.append(
        "- **Confirmation evidence:** a candidate that co-occurs with a documented extreme precipitation or dam-release event should be retained as a genuine extreme; "
        "isolated spikes with no corroborating evidence should be flagged as suspect and excluded from frequency analyses."
    )
    parts.append(
        "- **Documentation:** record all anomaly decisions in the quality-flag system with a reason code (e.g., 'confirmed extreme', 'instrument error', 'imputation artefact'); "
        "this ensures future users can understand the data provenance and apply appropriate analytical caveats."
    )
    return markdown.markdown('\n'.join(parts))

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

DARK_BG   = '#07111f'
GRID      = 'rgba(148,163,184,0.12)'
TEXT      = '#e5eefc'
SOFT      = '#9db0d1'
AXIS_FONT = dict(size=10, color=SOFT)


class QualityService:
    """Data-quality analytics: completeness heatmap, imputation summary,
    gap detection, anomaly-candidate review, and flag persistence."""

    def __init__(self, repository, data_dir: str | Path = 'data') -> None:
        self.repo = repository
        self.data_dir = Path(data_dir)
        self.flags_path = self.data_dir / 'quality_flags.json'

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_repo(self, station: str):
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if station in r.station_index), None)
        if station in getattr(self.repo, 'station_index', {}):
            return self.repo
        return None

    def _load_series(self, repo, station: str, feature: str) -> pd.Series:
        fd = repo.station_index[station]['feature_details'][feature]
        req = SeriesRequest(station=station, feature=feature,
                            start_date=fd['start_date'], end_date=fd['end_date'])
        df = repo.get_feature_series(req)
        ts = df.set_index('Timestamp')['Value'].sort_index()
        ts.index = pd.to_datetime(ts.index)
        return ts

    def _load_imputed_mask(self, repo, station: str, feature: str) -> pd.Series | None:
        """Return a boolean Series of imputed flags aligned to the series index."""
        df = repo.get_station_dataframe(station)
        imp_col = f'{feature}_imputed'
        if imp_col not in df.columns:
            return None
        mask = df.set_index('Timestamp')[imp_col].fillna('No').str.lower().eq('yes')
        mask.index = pd.to_datetime(mask.index)
        return mask

    def _axis_style(self) -> dict:
        return dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=AXIS_FONT)

    # ── 1. Completeness heatmap ───────────────────────────────────────────────

    def completeness(self, station: str, feature: str) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")
        if feature not in repo.station_index[station].get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

        ts = self._load_series(repo, station, feature)
        freq = repo.feature_frequency.get(feature, 'daily')
        unit = repo.feature_units.get(feature, '')

        if freq == 'monthly':
            # For monthly data: 1 obs expected per month
            monthly_obs = ts.resample('MS').count()
            pct_matrix = {}
            for dt, cnt in monthly_obs.items():
                yr, mo = dt.year, dt.month
                pct_matrix.setdefault(yr, {})[mo] = min(100.0, float(cnt) * 100)
        else:
            # Daily: expected = days in month
            monthly_obs = ts.resample('MS').count()
            pct_matrix = {}
            for dt, cnt in monthly_obs.items():
                yr, mo = dt.year, dt.month
                days_in_month = (pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(1)).day
                pct_matrix.setdefault(yr, {})[mo] = round(min(100.0, cnt / days_in_month * 100), 1)

        years = sorted(pct_matrix.keys())
        z = []
        text = []
        for yr in years:
            row_z, row_t = [], []
            for mo in range(1, 13):
                val = pct_matrix.get(yr, {}).get(mo, None)
                row_z.append(val if val is not None else -1)
                row_t.append(f'{val:.0f}%' if val is not None else 'No data')
            z.append(row_z)
            text.append(row_t)

        # Overall stats
        all_vals = [v for row in z for v in row if v >= 0]
        overall_pct = round(float(np.mean(all_vals)), 1) if all_vals else 0.0
        missing_months = sum(1 for row in z for v in row if v < 0)
        low_months = sum(1 for row in z for v in row if 0 <= v < 50)

        colorscale = [
            [0.0,  '#ef4444'],   # 0% — red
            [0.5,  '#f59e0b'],   # 50% — amber
            [0.75, '#84cc16'],   # 75% — lime
            [1.0,  '#22c55e'],   # 100% — green
        ]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=MONTH_ABBR,
            y=[str(yr) for yr in years],
            text=text,
            texttemplate='%{text}',
            colorscale=colorscale,
            zmin=0, zmax=100,
            colorbar=dict(
                title=dict(text='% available', font=dict(size=10, color=SOFT)),
                tickfont=dict(size=10, color=SOFT),
                thickness=12, len=0.7,
            ),
            hovertemplate='%{y} %{x}<br>Completeness: %{text}<extra></extra>',
        ))
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text=f'Data Completeness · {feature} · {station.replace("_", " ")}',
                       font=dict(size=13, color=TEXT), x=0, xanchor='left', pad=dict(l=4)),
            xaxis=dict(**self._axis_style(), side='top'),
            yaxis=dict(**self._axis_style(), autorange='reversed'),
        )

        return {
            'station': station, 'feature': feature, 'unit': unit,
            'years': years, 'overall_pct': overall_pct,
            'missing_months': missing_months, 'low_months': low_months,
            'total_months': len(all_vals) + missing_months,
            'figure': plotly.io.to_json(fig),
        }

    # ── 2. Imputation summary ────────────────────────────────────────────────

    def imputation_summary(self, dataset: str, feature: str | None = None) -> Dict[str, Any]:
        # Collect raw per-station-feature imputation stats across the right repo(s)
        repos = self.repo.repos if hasattr(self.repo, 'repos') else [self.repo]
        raw_rows: List[Dict] = []
        for repo in repos:
            if repo.dataset != dataset:
                continue
            for stn, meta in repo.station_index.items():
                for feat, fd in meta.get('feature_details', {}).items():
                    if feature and feat != feature:
                        continue
                    obs = fd.get('observations', 0)
                    imp = fd.get('imputed_points', 0)
                    total = obs
                    imp_pct = round(imp / total * 100, 2) if total > 0 else 0.0
                    raw_rows.append({
                        'station': stn,
                        'name': meta.get('name', stn).replace('_', ' '),
                        'feature': feat,
                        'observations': obs,
                        'imputed': imp,
                        'imp_pct': imp_pct,
                        'country': meta.get('country', ''),
                    })

        if not raw_rows:
            raise ValueError(f"No data found for dataset='{dataset}'" +
                             (f" feature='{feature}'" if feature else '') + '.')

        raw_df = pd.DataFrame(raw_rows)

        # For a specific feature, station ranking is naturally station-level already.
        # Without a feature filter, aggregate across all features so the ranking and
        # station counts genuinely refer to stations rather than station-feature rows.
        if feature:
            df = raw_df.sort_values('imp_pct', ascending=False).reset_index(drop=True)
        else:
            grouped = (
                raw_df.groupby(['station', 'name', 'country'], as_index=False)
                .agg(
                    observations=('observations', 'sum'),
                    imputed=('imputed', 'sum'),
                )
            )
            grouped['feature'] = 'all_features'
            grouped['imp_pct'] = np.where(
                grouped['observations'] > 0,
                grouped['imputed'] / grouped['observations'] * 100,
                0.0,
            ).round(2)
            df = grouped.sort_values('imp_pct', ascending=False).reset_index(drop=True)

        # Dataset-level stats
        total_obs = int(df['observations'].sum())
        total_imp = int(df['imputed'].sum())
        overall_imp_pct = round(total_imp / total_obs * 100, 2) if total_obs > 0 else 0.0
        stations_with_imputation = int((df['imp_pct'] > 0).sum())
        high_imputation = int((df['imp_pct'] >= 20).sum())

        # Top 20 most-imputed stations for bar chart
        top = df.head(20)
        bar_colors = ['#ef4444' if v >= 20 else '#f59e0b' if v >= 5 else '#38bdf8'
                      for v in top['imp_pct']]

        fig = go.Figure(go.Bar(
            x=top['imp_pct'].tolist(),
            y=top['name'].tolist(),
            orientation='h',
            marker_color=bar_colors,
            hovertemplate='%{y}<br>Imputation: %{x:.1f}%<extra></extra>',
        ))
        feat_label = feature or 'all features'
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            margin=dict(l=10, r=20, t=40, b=10),
            title=dict(text=f'Top stations by imputation rate · {dataset} · {feat_label}',
                       font=dict(size=13, color=TEXT), x=0, xanchor='left', pad=dict(l=4)),
            xaxis=dict(**self._axis_style(), title='Imputation %'),
            yaxis=dict(**self._axis_style(), autorange='reversed'),
            height=max(300, len(top) * 22 + 80),
        )

        return {
            'dataset': dataset,
            'feature': feature,
            'total_observations': total_obs,
            'total_imputed': total_imp,
            'overall_imp_pct': overall_imp_pct,
            'stations_with_imputation': stations_with_imputation,
            'high_imputation_stations': high_imputation,
            'total_stations': len(df['station'].unique()),
            'rows': df[['station', 'name', 'feature', 'observations', 'imputed', 'imp_pct', 'country']].to_dict('records'),
            'figure': plotly.io.to_json(fig),
        }

    # ── 3. Gap detection ────────────────────────────────────────────────────

    def gaps(self, station: str, feature: str) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")
        if feature not in repo.station_index[station].get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

        ts = self._load_series(repo, station, feature)
        freq = repo.feature_frequency.get(feature, 'daily')
        unit = repo.feature_units.get(feature, '')
        pd_freq = 'MS' if freq == 'monthly' else 'D'
        unit_label = 'months' if freq == 'monthly' else 'days'

        # Reindex to full regular grid
        full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq=pd_freq)
        ts_full = ts.reindex(full_idx)

        null_mask = ts_full.isna()
        gap_list: List[Dict] = []

        in_gap = False
        gap_start = None
        for dt, is_null in null_mask.items():
            if is_null and not in_gap:
                in_gap = True
                gap_start = dt
            elif not is_null and in_gap:
                in_gap = False
                length = int((dt - gap_start).days) if freq != 'monthly' else \
                    int((dt.year - gap_start.year) * 12 + dt.month - gap_start.month)
                if freq == 'monthly':
                    severity = ('major' if length >= 12 else
                                'moderate' if length >= 3 else 'minor')
                    gap_end = dt - pd.offsets.MonthBegin(1)
                else:
                    severity = ('major' if length >= 30 else
                                'moderate' if length >= 7 else 'minor')
                    gap_end = dt - pd.Timedelta(days=1)
                gap_list.append({
                    'start': str(gap_start.date()),
                    'end': str(gap_end.date()),
                    'length': length,
                    'unit': unit_label,
                    'severity': severity,
                })

        # If series ends with a gap
        if in_gap:
            length = int((ts_full.index[-1] - gap_start).days) if freq != 'monthly' else \
                int((ts_full.index[-1].year - gap_start.year) * 12 +
                    ts_full.index[-1].month - gap_start.month) + 1
            if freq == 'monthly':
                severity = ('major' if length >= 12 else 'moderate' if length >= 3 else 'minor')
            else:
                severity = ('major' if length >= 30 else 'moderate' if length >= 7 else 'minor')
            gap_list.append({
                'start': str(gap_start.date()),
                'end': str(ts_full.index[-1].date()),
                'length': length,
                'unit': unit_label,
                'severity': severity,
            })

        gap_list.sort(key=lambda g: g['length'], reverse=True)

        total_missing = int(null_mask.sum())
        total_pts = len(full_idx)
        missing_pct = round(total_missing / total_pts * 100, 2) if total_pts else 0.0

        # Timeline figure: series with gap regions shaded
        display = ts_full.tail(min(len(ts_full), 2000))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display.index, y=display.values,
            mode='lines', line=dict(color='#38bdf8', width=1.5),
            name=feature, connectgaps=False,
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f} ' + unit + '<extra></extra>',
        ))
        # Shade top-5 largest gaps
        for g in gap_list[:5]:
            color = ('#ef444430' if g['severity'] == 'major' else
                     '#f59e0b25' if g['severity'] == 'moderate' else '#64748b20')
            fig.add_vrect(x0=g['start'], x1=g['end'],
                          fillcolor=color, line_width=0, layer='below')

        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text=f'Gap Detection · {feature} · {station.replace("_", " ")}',
                       font=dict(size=13, color=TEXT), x=0, xanchor='left', pad=dict(l=4)),
            xaxis=dict(**self._axis_style()),
            yaxis=dict(**self._axis_style(), title=unit or feature),
            showlegend=False,
        )

        return {
            'station': station, 'feature': feature,
            'total_points': total_pts, 'missing_points': total_missing,
            'missing_pct': missing_pct,
            'gap_count': len(gap_list),
            'major': sum(1 for g in gap_list if g['severity'] == 'major'),
            'moderate': sum(1 for g in gap_list if g['severity'] == 'moderate'),
            'minor': sum(1 for g in gap_list if g['severity'] == 'minor'),
            'gaps': gap_list[:50],   # cap at 50 for payload size
            'figure': plotly.io.to_json(fig),
        }

    # ── 4. Anomaly candidates ────────────────────────────────────────────────

    @staticmethod
    def _modified_zscore(values: np.ndarray) -> np.ndarray:
        """
        Iglewicz & Hoaglin (1993) modified z-score using Median Absolute Deviation.

        M_i = 0.6745 * (x_i - median(x)) / MAD
        where MAD = median(|x_i - median(x)|).

        Preferred over the standard z-score for hydrological data, which is
        right-skewed and non-normal.  The standard z-score is distorted by the
        very outliers it is trying to detect; MAD-based detection is robust.
        Recommended threshold: |M_i| > 3.5  (Iglewicz & Hoaglin 1993).
        """
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad == 0:
            # Fallback: use mean absolute deviation if MAD is zero
            mad = float(np.mean(np.abs(values - med))) or 1e-10
        return 0.6745 * (values - med) / mad

    def anomaly_candidates(self, station: str, feature: str, z_thresh: float = 3.5) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")
        if feature not in repo.station_index[station].get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

        ts = self._load_series(repo, station, feature)
        unit = repo.feature_units.get(feature, '')
        imp_mask = self._load_imputed_mask(repo, station, feature)

        values_arr = ts.values.astype(float)
        median_val = float(np.median(values_arr))
        mad_val    = float(np.median(np.abs(values_arr - median_val)))
        if mad_val == 0:
            mad_val = float(np.mean(np.abs(values_arr - median_val))) or 1e-10

        # Modified z-scores (Iglewicz & Hoaglin 1993) — robust to skewness
        mod_z_arr = self._modified_zscore(values_arr)
        mod_z_series = pd.Series(np.abs(mod_z_arr), index=ts.index)
        anomaly_idx = mod_z_series[mod_z_series >= z_thresh].index

        candidates = []
        existing_flags = self._load_flags().get(station, {}).get(feature, {})

        for dt in anomaly_idx:
            val  = float(ts.loc[dt])
            mz   = float(mod_z_series.loc[dt])
            pos  = ts.index.get_loc(dt)
            ctx_before = ts.iloc[max(0, pos - 3): pos].tolist()
            ctx_after  = ts.iloc[pos + 1: pos + 4].tolist()
            is_imputed = bool(imp_mask.loc[dt]) if (imp_mask is not None and dt in imp_mask.index) else False
            date_str = str(dt.date())
            candidates.append({
                'date': date_str,
                'value': round(val, 4),
                'z_score': round(mz, 2),        # field name kept for API compat
                'above_mean': val > median_val,
                'is_imputed': is_imputed,
                'context_before': [round(v, 4) for v in ctx_before],
                'context_after':  [round(v, 4) for v in ctx_after],
                'flag': existing_flags.get(date_str, 'none'),
            })

        # Sort by modified z-score descending
        candidates.sort(key=lambda c: c['z_score'], reverse=True)

        # Build time-series figure with anomalies highlighted
        anomaly_dates = {c['date'] for c in candidates}
        normal_mask = pd.Series([str(d.date()) not in anomaly_dates for d in ts.index], index=ts.index)
        anomaly_mask = ~normal_mask

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index[normal_mask], y=ts.values[normal_mask],
            mode='lines', name='Normal', line=dict(color='#4A90D9', width=1.2),
        ))
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=ts.index[anomaly_mask], y=ts.values[anomaly_mask],
                mode='markers', name=f'Anomaly (|M| ≥ {z_thresh})',
                marker=dict(color='#E74C3C', size=7, symbol='circle'),
            ))
        fig.update_layout(
            title=f'Anomaly Candidates — {station.replace("_", " ")} · {feature}',
            xaxis_title='Date', yaxis_title=f'{feature} ({unit})',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=60, b=40, l=60, r=20),
            hovermode='x unified',
        )

        return {
            'station': station, 'feature': feature, 'unit': unit,
            'z_thresh': z_thresh,
            'mean': round(median_val, 4),
            'std':  round(mad_val, 4),
            'detection_method': 'modified_zscore_MAD',
            'detection_note': (
                'Outlier detection uses the modified z-score (Iglewicz & Hoaglin 1993) '
                'based on median and MAD, which is robust to skewness and to the '
                'influence of outliers on the detection threshold. '
                f'Threshold: |M| ≥ {z_thresh}.'
            ),
            'total': len(candidates),
            'unflagged': sum(1 for c in candidates if c['flag'] == 'none'),
            'candidates': candidates[:100],
            'figure': plotly.io.to_json(fig),
        }

    # ── 5. Flag persistence ─────────────────────────────────────────────────

    def _load_flags(self) -> Dict:
        if self.flags_path.exists():
            try:
                return json.loads(self.flags_path.read_text())
            except Exception:
                pass
        return {}

    def save_flag(self, station: str, feature: str, date_str: str, flag: str) -> Dict[str, Any]:
        """flag must be one of: real, sensor_error, uncertain, none (clears)"""
        allowed = {'real', 'sensor_error', 'uncertain', 'none'}
        if flag not in allowed:
            raise ValueError(f"flag must be one of {allowed}")
        flags = self._load_flags()
        flags.setdefault(station, {}).setdefault(feature, {})
        if flag == 'none':
            flags[station][feature].pop(date_str, None)
        else:
            flags[station][feature][date_str] = flag
        self.flags_path.write_text(json.dumps(flags, indent=2))
        return {'station': station, 'feature': feature, 'date': date_str, 'flag': flag}

    def flag_summary(self) -> Dict[str, Any]:
        flags = self._load_flags()
        total = 0
        breakdown: Dict[str, int] = {'real': 0, 'sensor_error': 0, 'uncertain': 0}
        details = []
        for stn, feats in flags.items():
            for feat, dates in feats.items():
                for d, f in dates.items():
                    total += 1
                    breakdown[f] = breakdown.get(f, 0) + 1
                    details.append({'station': stn, 'feature': feat, 'date': d, 'flag': f})
        details.sort(key=lambda x: x['date'], reverse=True)
        return {'total': total, 'breakdown': breakdown, 'recent': details[:20]}
