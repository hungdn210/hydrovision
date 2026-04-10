"""
network_service.py
~~~~~~~~~~~~~~~~~~
Spatial Network Analysis for the Mekong basin.

Topology: documented Mekong/Lower Mekong Basin river system — main-stem order
and tributary connections are hardcoded and clearly labelled.  Sub-basin GeoJSON
is only a single boundary polygon, so connections are derived from published
basin reports and station coordinates rather than automated spatial joins.

Travel times: empirical cross-correlation of discharge series, reported in days.
Contribution: discharge-ratio proxy (labelled as estimate, not hydraulic routing).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io

from .analysis_service import _gemini_generate
from .data_loader import DataRepository, MultiDataRepository, SeriesRequest
from .figure_theme import (
    DARK_BG, TEXT, SOFT, GRID_LIGHT, GRID_STRONG,
)


def _generate_network_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    fallback = _fallback_network_analysis(result)
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return fallback
    try:
        prompt = _network_prompt(result)
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_network_analysis(result)


def _network_prompt(result: Dict[str, Any]) -> str:
    travel = result.get('travel_times') or []
    sample = '\n'.join(
        f"- {row['upstream_name'].replace('_', ' ')} -> {row['downstream_name'].replace('_', ' ')}: "
        f"{row['lag_days']} day lag, corr={row['correlation']:.3f}, overlap={row['overlap_count']} {row.get('overlap_unit', 'steps')}"
        for row in travel[:10]
    ) or 'No travel-time pairs available.'
    main_stem = ', '.join(s.replace('_', ' ') for s in result.get('main_stem', [])) or 'Unavailable'
    n_pairs = len(travel)
    if travel:
        avg_lag = sum(r.get('lag_days') or 0 for r in travel) / n_pairs
        avg_corr = sum(r.get('correlation', 0) for r in travel) / n_pairs
    else:
        avg_lag = avg_corr = 0
    return (
        'Act as a professional hydrologist writing a detailed technical interpretation for a river-network topology dashboard.\n\n'
        'RESPONSE FORMAT (STRICT — use markdown, no text outside these headings):\n\n'
        '## Network Summary\n'
        'Write 4–5 sentences. Describe the scale of the network, how it represents the basin, the relationship between node count and edge count, '
        'and whether the empirical travel-time evidence is strong or limited. Conclude with one sentence on the operational value of this topology view.\n\n'
        '## Connectivity Structure\n'
        'Exactly 5 bullet points covering: total nodes and edges; main-stem coverage and its role as the backbone; '
        'tributary complexity (branching factor); density of the directed graph relative to a fully connected network; '
        'and one note on what structural gaps (missing tributaries, ungauged reaches) imply for analysis completeness.\n\n'
        '## Travel-Time Interpretation\n'
        'Exactly 5 bullet points: cite the number of computed travel-time pairs; '
        'name the strongest correlated pair and interpret its hydrological significance; '
        'cite the longest inferred lag pair and explain what that travel time implies; '
        'comment on the average lag and average correlation quality; '
        'note one caveat about using cross-correlation lags as a proxy for hydraulic travel times.\n\n'
        '## Methodology & Limitations\n'
        'Exactly 4 bullet points: how the network topology was derived (hardcoded from published basin reports); '
        'how empirical travel times are estimated (AR(1) pre-whitened cross-correlation); '
        'what assumptions underlie the contribution proxy (discharge ratio, not hydraulic routing); '
        'and what the key uncertainty sources are (gauge placement, unmeasured tributaries, flow regulation).\n\n'
        '## Operational Relevance\n'
        'Exactly 3 bullet points: (1) how network analysts should use the topology for monitoring; '
        '(2) how travel-time estimates can guide forecast lead-times or flood early warning; '
        '(3) one important caveat to communicate when presenting network results to non-specialist stakeholders.\n\n'
        'RULES:\n'
        '- Cite specific counts, lag values, and correlations from the data.\n'
        '- Replace underscores with spaces.\n'
        '- Keep tone professional and technical throughout.\n'
        '- Do not describe this as a hydraulic routing model.\n\n'
        f"Dataset: {result.get('dataset', 'mekong')}\n"
        f"Node count: {result.get('node_count')}\n"
        f"Edge count: {result.get('edge_count')}\n"
        f"Main stem stations: {main_stem}\n"
        f"Travel-time pairs computed: {n_pairs}\n"
        f"Average lag across all pairs: {avg_lag:.1f} days\n"
        f"Average correlation across all pairs: {avg_corr:.3f}\n"
        f"Topology note: {result.get('note', '')}\n"
        f"Travel-time sample:\n{sample}\n"
    )


def _fallback_network_analysis(result: Dict[str, Any], note: str | None = None) -> str:
    travel = result.get('travel_times') or []
    dataset = result.get('dataset', 'mekong').replace('_', ' ')
    node_count = result.get('node_count', 0)
    edge_count = result.get('edge_count', 0)
    main_stem = result.get('main_stem', [])
    n_pairs = len(travel)

    parts = ['## Network Summary']
    intro = (
        f"The **{dataset}** river network comprises **{node_count} gauging stations** connected by "
        f"**{edge_count} directed edges**, representing the documented upstream-downstream routing structure of the basin. "
        f"The topology includes {len(main_stem)} main-stem stations spanning from the upper basin to the delta, "
        f"with additional tributary branches feeding into the main stem at multiple confluence points. "
        f"Travel-time estimates are derived from empirical cross-correlation of discharge series between adjacent station pairs, "
        f"providing an observational basis for understanding basin-scale flow propagation timing."
    )
    if note:
        intro += ' ' + note
    parts.append(intro)

    parts.append('## Connectivity Structure')
    parts.append(
        f"- **Network scale:** {node_count} nodes and {edge_count} directed edges represent the gauged portion of the basin; "
        f"ungauged tributaries and headwater reaches are not represented in this view."
    )
    parts.append(
        f"- **Main-stem backbone:** {len(main_stem)} stations trace the primary longitudinal flow pathway from upper basin to delta, "
        f"forming the core monitoring spine of the network."
    )
    if edge_count > 0 and node_count > 1:
        density = round(edge_count / (node_count * (node_count - 1)) * 100, 1)
        parts.append(
            f"- **Network density:** with {edge_count} edges in a {node_count}-node directed graph, "
            f"the connectivity density is approximately {density}%, consistent with a sparse hierarchical river topology rather than a fully connected network."
        )
    else:
        parts.append("- **Network density:** insufficient node/edge data for density computation.")
    parts.append(
        "- **Tributary branching:** multiple tributary sub-systems feed the main stem, reflecting the compound catchment structure "
        "typical of large international river basins with distributed headwater regions."
    )
    parts.append(
        "- **Structural gaps:** some stations may lack connectivity assignments where gauge records are too short for cross-correlation "
        "or where tributaries are outside the documented topology; these should be treated as data voids rather than hydraulic disconnections."
    )

    parts.append('## Travel-Time Interpretation')
    if travel:
        strongest = max(travel, key=lambda row: row.get('correlation', 0.0))
        longest = max(travel, key=lambda row: row.get('lag_days') or 0)
        avg_lag = sum(r.get('lag_days') or 0 for r in travel) / n_pairs
        avg_corr = sum(r.get('correlation', 0) for r in travel) / n_pairs
        parts.append(
            f"- **Travel-time coverage:** {n_pairs} station pair(s) have sufficient overlapping discharge records "
            f"for empirical cross-correlation lag estimation."
        )
        parts.append(
            f"- **Strongest cross-correlation:** {strongest['upstream_name'].replace('_', ' ')} to "
            f"{strongest['downstream_name'].replace('_', ' ')} achieves a correlation of {strongest['correlation']:.3f}, "
            f"indicating a statistically coherent discharge signal with a lag of {strongest.get('lag_days')} day(s)."
        )
        parts.append(
            f"- **Longest inferred lag:** {longest['upstream_name'].replace('_', ' ')} to "
            f"{longest['downstream_name'].replace('_', ' ')} shows a propagation time of {longest.get('lag_days')} day(s), "
            f"reflecting the extended travel path through the lower basin or delta distributaries."
        )
        parts.append(
            f"- **Average characteristics:** across all computed pairs, the mean travel lag is {avg_lag:.1f} day(s) "
            f"with a mean correlation of {avg_corr:.3f}, suggesting {'generally coherent' if avg_corr > 0.5 else 'moderately coherent'} discharge propagation signals across the network."
        )
        parts.append(
            "- **Cross-correlation caveat:** empirical lags are computed from AR(1) pre-whitened series to reduce autocorrelation bias. "
            "They represent statistical timing relationships, not hydraulic residence times; actual flood wave propagation will vary with flow magnitude, channel morphology, and regulation."
        )
    else:
        parts.append("- **Travel-time evidence:** insufficient overlapping discharge data for cross-correlation lag estimation.")
        parts.append("- **Recommendation:** extend the common overlap period or reduce the lookback requirement to enable lag estimation for more station pairs.")
        parts.append("- **Alternative:** where cross-correlation fails, published MRC or basin-authority travel-time estimates can serve as proxies for operational timing.")
        parts.append("- **Impact on analysis:** absence of empirical lags limits the ability to validate upstream flood signal propagation timing in this view.")
        parts.append("- **Caveat:** even where correlations are available, they reflect statistical relationships under historical average conditions and may not hold during extreme events.")

    parts.append('## Methodology & Limitations')
    parts.append(
        "- **Topology source:** the directed graph is hardcoded from published MRC basin reports and station coordinates verified against peer-reviewed hydrological literature. "
        "Uncertain tributary assignments are marked in the source and should be treated as approximate."
    )
    parts.append(
        "- **Travel-time method:** empirical lags are estimated using AR(1) pre-whitened cross-correlation of daily discharge series. "
        "Pre-whitening removes serial autocorrelation to prevent spurious lag detection; the lag at maximum positive correlation is reported."
    )
    parts.append(
        "- **Contribution proxy:** upstream discharge contributions are estimated as the ratio of mean upstream station discharge to mean target station discharge. "
        "This is a hydrological proxy, not a hydraulic routing result — it does not account for channel losses, evaporation, or parallel unmeasured tributaries."
    )
    parts.append(
        "- **Key uncertainties:** gauge placement relative to actual confluences, regulation effects from dams and diversions, and the absence of ungauged sub-basin contributions "
        "are the principal sources of uncertainty in both the topology and the travel-time estimates."
    )

    parts.append('## Operational Relevance')
    parts.append(
        "- **Monitoring application:** the topology view is most useful for explaining upstream-downstream connectivity, identifying monitoring gaps, "
        "and assessing which stations provide redundant versus unique basin coverage."
    )
    parts.append(
        "- **Flood early-warning use:** empirical lag estimates can inform lead-time expectations for flood wave propagation between gauging stations, "
        "provided the operational team understands these are statistical guides rather than deterministic hydraulic predictions."
    )
    parts.append(
        "- **Communication note:** when presenting network results to non-specialist stakeholders, emphasise that tributary assignments and travel times represent documented basin knowledge "
        "and statistical evidence, not a real-time hydraulic model output."
    )
    return markdown.markdown('\n'.join(parts))


def _generate_contrib_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    fallback = _fallback_contrib_analysis(result)
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return fallback
    try:
        prompt = _contrib_prompt(result)
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_contrib_analysis(result)


def _contrib_prompt(result: Dict[str, Any]) -> str:
    target = result.get('target_name', 'Unknown')
    target_q = result.get('target_mean_q', 'Unknown')
    rows = result.get('rows', [])
    top_contrib = '\n'.join(
        f"- {r['name'].replace('_', ' ')}: {r['contrib_pct']}% ({r['mean_q']} m³/s)"
        for r in rows[:8]
    ) or 'No upstream contributions available.'
    top3_pct = sum(r.get('contrib_pct', 0) for r in rows[:3])
    upstream_count = result.get('upstream_count', 0)

    return (
        f'Act as a professional hydrologist writing a detailed interpretation of upstream discharge contributions to {target.replace("_", " ")}.\n\n'
        'RESPONSE FORMAT (STRICT — markdown, no text outside these headings):\n\n'
        '## Contribution Summary\n'
        'Write 4–5 sentences. State the target station and its mean discharge, how many upstream stations are tracked, '
        'name the top 1–2 contributors and their percentages, characterise whether the contribution structure is highly concentrated or distributed, '
        'and note what this implies for flow generation mechanisms in the basin.\n\n'
        '## Dominant Upstream Sources\n'
        'Exactly 5 bullet points. For each of the top contributors: name the station, cite the contribution percentage and mean discharge in m³/s, '
        'interpret the hydrological significance (e.g., large regulated headwater, major tributary confluence, unregulated mountain catchment), '
        'and note any caveats about the reliability of that particular estimate.\n\n'
        '## Hydrological Significance\n'
        'Exactly 4 bullet points: (1) concentration index — what share of total discharge is explained by the top 3 contributors; '
        '(2) what a concentrated vs distributed structure implies for flood risk at the target; '
        '(3) seasonal variability — how monsoon or snowmelt cycles might shift contribution rankings; '
        '(4) implications for water resource management or hydropower scheduling at the target station.\n\n'
        '## Methodological Context\n'
        'Exactly 3 bullet points: (1) explain the discharge-ratio proxy method and its formula; '
        '(2) list specific processes not captured (channel losses, evapotranspiration, unmeasured tributaries, delta distributaries); '
        '(3) recommend how an analyst should interpret or supplement this proxy in operational practice.\n\n'
        '## Operational Relevance\n'
        'Exactly 3 bullet points: (1) how contribution rankings can guide monitoring prioritisation; '
        '(2) how this analysis supports flood early-warning decisions at the target station; '
        '(3) one important limitation to communicate to water managers or non-specialist users.\n\n'
        'RULES:\n'
        '- Cite all percentage and discharge values from the provided data.\n'
        '- Replace underscores with spaces.\n'
        '- Professional and technical tone throughout.\n'
        '- Do not describe this as a hydraulic routing model.\n\n'
        f"Target Station: {target.replace('_', ' ')}\n"
        f"Target Mean Discharge: {target_q} m³/s\n"
        f"Upstream Station Count: {upstream_count}\n"
        f"Top-3 combined contribution: ~{top3_pct:.0f}%\n"
        f"All Contributors (ranked):\n{top_contrib}\n"
    )


def _fallback_contrib_analysis(result: Dict[str, Any], note: str | None = None) -> str:
    target = result.get('target_name', 'the target station').replace('_', ' ')
    target_q = result.get('target_mean_q', 'N/A')
    rows = result.get('rows', [])
    upstream_count = result.get('upstream_count', len(rows))

    top3_pct = sum(r.get('contrib_pct', 0) for r in rows[:3])
    concentrated = top3_pct >= 70

    parts = ['## Contribution Summary']
    intro = (
        f"Upstream discharge contribution analysis for **{target}** tracks {upstream_count} upstream station(s) "
        f"using a discharge-ratio proxy referenced to the target mean discharge of {target_q} m³/s. "
    )
    if rows:
        top = rows[0]
        intro += (
            f"The dominant upstream source is **{top['name'].replace('_', ' ')}**, contributing approximately {top['contrib_pct']}% "
            f"of the target flow with a mean discharge of {top['mean_q']} m³/s. "
        )
        if len(rows) >= 3:
            intro += (
                f"The top 3 contributors together account for approximately {top3_pct:.0f}% of tracked upstream flow, "
                f"indicating a {'highly concentrated' if concentrated else 'relatively distributed'} contribution structure."
            )
    if note:
        intro += ' ' + note
    parts.append(intro)

    parts.append('## Dominant Upstream Sources')
    if rows:
        for r in rows[:5]:
            parts.append(
                f"- **{r['name'].replace('_', ' ')}:** {r['contrib_pct']}% contribution, mean discharge {r['mean_q']} m³/s — "
                f"{'primary driver of target station flow regime' if r == rows[0] else 'significant upstream source contributing to base and event flows'}."
            )
        if not rows:
            parts.append("- **No upstream sources found** within the documented topology for this station.")
    else:
        parts.append("- **No upstream sources found** within the documented network topology for this station.")

    parts.append('## Hydrological Significance')
    if rows:
        parts.append(
            f"- **Concentration index:** the top 3 contributors account for approximately {top3_pct:.0f}% of tracked upstream discharge, "
            f"{'suggesting that a small number of sub-catchments dominate the target flow regime' if concentrated else 'indicating a distributed flow generation pattern across multiple sub-catchments'}."
        )
    else:
        parts.append("- **Concentration index:** no upstream contribution data available for analysis.")
    parts.append(
        "- **Flood risk implication:** a highly concentrated contribution structure means that extreme events in the dominant upstream catchment(s) "
        "will disproportionately drive flood peaks at the target station; distributed structures are generally associated with more attenuated flood responses."
    )
    parts.append(
        "- **Seasonal variability:** monsoon intensification, snowmelt pulses, or ENSO-driven anomalies may shift the relative contribution rankings "
        "seasonally — the discharge-ratio proxy reflects long-run mean conditions and does not capture within-year variability."
    )
    parts.append(
        "- **Water resource management:** understanding dominant upstream contributors supports reservoir scheduling, hydropower dispatch optimisation, "
        "and early-warning system design by identifying which upstream stations carry the most predictive information for the target."
    )

    parts.append('## Methodological Context')
    parts.append(
        "- **Discharge-ratio proxy:** contribution percentages are computed as the ratio of each upstream station's long-run mean discharge "
        "to the target station's long-run mean discharge. This is a first-order hydrological estimate, not a hydraulic routing result."
    )
    parts.append(
        "- **Processes not captured:** channel transmission losses, evapotranspiration from floodplains, groundwater exchange, "
        "diversion abstractions, and contributions from unmeasured tributaries are all absent from this proxy; "
        "the actual hydrological partitioning in the field may differ substantially."
    )
    parts.append(
        "- **Operational use:** treat contribution estimates as indicative rankings rather than precise allocation percentages. "
        "Supplement with catchment area ratios, hydrograph shape comparison, or distributed hydrological model output for higher-confidence attribution."
    )

    parts.append('## Operational Relevance')
    parts.append(
        "- **Monitoring prioritisation:** stations with high contribution percentages should be treated as high-value upstream sentinels; "
        "ensuring their data quality and transmission reliability is critical for downstream forecasting accuracy."
    )
    parts.append(
        "- **Early-warning use:** when upstream sentinel stations enter Flood Risk or Flood Watch status (see risk map), "
        "the contribution ranking provides an estimate of how strongly that signal will propagate to the target station."
    )
    parts.append(
        "- **Communication caveat:** when sharing contribution figures with water managers, clearly label the values as discharge-ratio proxies "
        "and not hydraulic routing outputs; over-interpreting the precision of these percentages can lead to misplaced operational confidence."
    )
    return markdown.markdown('\n'.join(parts))


# ── Mekong river topology ─────────────────────────────────────────────────────
# (upstream_station, downstream_station) tuples.
# Source: MRC published basin maps + station coordinates verified against
# published literature.  Uncertain assignments are marked with '?'.

MEKONG_EDGES: List[Tuple[str, str]] = [
    # ── Main Mekong stem (Jinghong → delta) ──────────────────────────────────
    ('Jinghong',         'Chiang_Saen'),
    ('Chiang_Saen',      'Chiang_Khan'),
    ('Chiang_Khan',      'Luang_Prabang'),
    ('Luang_Prabang',    'Ban_Pak_Kanhoung'),
    ('Ban_Pak_Kanhoung', 'Vientiane_KM4'),
    ('Vientiane_KM4',    'Nong_Khai'),
    ('Nong_Khai',        'Nakhon_Phanom'),
    ('Nakhon_Phanom',    'Mukdahan'),
    ('Mukdahan',         'Ban_Kengdone'),
    ('Ban_Kengdone',     'Khong_Chiam'),
    ('Khong_Chiam',      'Pakse'),
    ('Pakse',            'Stung_Treng'),
    ('Stung_Treng',      'Kratie'),
    ('Kratie',           'Kompong_Cham'),
    ('Kompong_Cham',     'Chaktomuk'),
    ('Chaktomuk',        'Phnom_Penh_Port'),
    ('Phnom_Penh_Port',  'Chroy_Chang_Var'),
    # Phnom Penh splits: Tien (Mekong) and Bassac (Hau) branches
    ('Chroy_Chang_Var',  'Tan_Chau'),
    ('Chroy_Chang_Var',  'Chau_Doc'),
    ('Tan_Chau',         'My_Tho'),

    # ── Kok River (northern Thailand → Chiang Saen) ───────────────────────────
    ('Ban_Tha_Ton',       'Ban_Tha_Mai_Liam'),
    ('Ban_Tha_Mai_Liam',  'Ban_Huai_Yano_Mai'),
    ('Ban_Huai_Yano_Mai', 'Chiang_Saen'),

    # ── Nam Ngum / central Laos tributaries → Vientiane ──────────────────────
    ('Ban_Na_Luang',      'Vientiane_KM4'),

    # ── Loei River → joins Mekong near Nong Khai ─────────────────────────────
    ('Ban_Pak_Huai',      'Nong_Khai'),

    # ── Nam Mun River system (NE Thailand) → Khong Chiam ─────────────────────
    ('Ban_Huai_Khayuong', 'Yasothon'),
    ('Ban_Nong_Kiang',    'Ban_Tad_Ton'),
    ('Ban_Tad_Ton',       'Ban_Chot'),
    ('Ban_Chot',          'Khong_Chiam'),
    ('Yasothon',          'Khong_Chiam'),

    # ── Se Banghiang / lower Laos tributaries → Stung Treng ──────────────────
    ('Chantangoy',        'Ban_Kamphun'),
    ('Ban_Kamphun',       'Stung_Treng'),

    # ── Sesan River (Vietnam central highlands → Cambodia) ───────────────────
    ('Dak_To',            'Kontum'),
    ('Pleiku',            'Kontum'),
    ('Kontum',            'Lumphat'),
    ('Lumphat',           'Stung_Treng'),

    # ── Srepok River (Vietnam central highlands → Cambodia) ──────────────────
    ('Buon_Ho',           'Buon_Me_Thuoc'),
    ('Cau_14_(Buon_Bur)', 'Buon_Me_Thuoc'),
    ('Buon_Me_Thuoc',     'Dak_Nong'),
    ('Duc_Xuyen',         'Dak_Nong'),
    ('Giang_Son',         'Dak_Nong'),
    ('Dak_Nong',          'Ban_Don'),
    ('Ban_Don',           'Stung_Treng'),

    # ── Tonle Sap (Cambodia great lake) → Phnom Penh ─────────────────────────
    ('Battambang',        'Phnom_Penh_Port'),

    # ── Bassac / Hau branch delta ─────────────────────────────────────────────
    ('Chau_Doc',          'Can_Tho'),
    ('Chau_Doc',          'Vam_Nao'),
    ('Vam_Nao',           'My_Thuan'),
    ('Can_Tho',           'Vi_Thanh'),
    ('Can_Tho',           'Phung_Hiep'),
    ('Phung_Hiep',        'Ca_Mau'),
    ('Can_Tho',           'Dai_Ngai'),

    # ── Tien / Mekong branch delta ────────────────────────────────────────────
    ('My_Tho',            'My_Thuan'),
    ('My_Tho',            'Cai_Be'),
    ('My_Tho',            'Cai_Lay'),
    ('My_Tho',            'Cho_Lach'),
    ('My_Tho',            'Cho_Moi'),
    ('My_Tho',            'Vam_Kenh'),
    ('Cho_Moi',           'Tan_Hiep'),
    ('Cho_Lach',          'Batri'),
    ('Batri',             'Dinh_An'),
    ('My_Tho',            'Ben_Luc'),
]

# Stations confirmed on the main Mekong stem, ordered upstream→downstream
MEKONG_MAIN_STEM: List[str] = [
    'Jinghong', 'Chiang_Saen', 'Chiang_Khan', 'Luang_Prabang',
    'Ban_Pak_Kanhoung', 'Vientiane_KM4', 'Nong_Khai', 'Nakhon_Phanom',
    'Mukdahan', 'Ban_Kengdone', 'Khong_Chiam', 'Pakse',
    'Stung_Treng', 'Kratie', 'Kompong_Cham', 'Chaktomuk',
    'Phnom_Penh_Port', 'Chroy_Chang_Var', 'Tan_Chau', 'My_Tho',
]


class NetworkService:
    """River network topology, travel time, and Plotly visualization."""

    MAX_LAG_DAYS = 180
    MAX_LAG_MONTHS = 6
    MIN_OVERLAP_DAYS = 365
    MIN_OVERLAP_MONTHS = 24

    def __init__(self, repository: MultiDataRepository) -> None:
        self.repository = repository

    # ── helpers ───────────────────────────────────────────────────────────────

    def _repo_for(self, dataset: str) -> DataRepository:
        for r in self.repository.repos:
            if r.dataset == dataset:
                return r
        raise ValueError(f'Unknown dataset: {dataset!r}')

    def _valid_edges(self, repo: DataRepository) -> List[Tuple[str, str]]:
        s = set(repo.station_index.keys())
        return [(u, d) for u, d in MEKONG_EDGES if u in s and d in s]

    def _build_adjacency(
        self, repo: DataRepository
    ) -> Dict[str, Dict[str, List[str]]]:
        adj: Dict[str, Dict[str, List[str]]] = {
            s: {'upstream': [], 'downstream': []}
            for s in repo.station_index
        }
        for u, d in self._valid_edges(repo):
            adj[u]['downstream'].append(d)
            adj[d]['upstream'].append(u)
        return adj

    def _all_upstream(
        self, station: str, adj: Dict[str, Dict[str, List[str]]]
    ) -> Set[str]:
        """BFS all ancestors of station in the topology DAG."""
        visited: Set[str] = set()
        queue = list(adj.get(station, {}).get('upstream', []))
        while queue:
            s = queue.pop()
            if s not in visited:
                visited.add(s)
                queue.extend(adj.get(s, {}).get('upstream', []))
        return visited

    def _discharge_series(
        self, repo: DataRepository, station: str
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        fd = repo.station_index[station]['feature_details'].get('Discharge')
        if fd is None:
            return None, None
        try:
            req = SeriesRequest(
                station=station, feature='Discharge',
                start_date=fd['start_date'], end_date=fd['end_date'],
            )
            df = repo.get_feature_series(req)
            ts = df.set_index('Timestamp')['Value'].sort_index()
            freq = repo.feature_frequency.get('Discharge', 'daily')
            if freq == 'monthly':
                return ts.resample('MS').mean().dropna(), 'monthly'
            return ts.resample('D').mean().dropna(), 'daily'
        except Exception:
            return None, None

    # ── travel time via cross-correlation ────────────────────────────────────

    # Minimum CCF peak correlation below which lag estimate is considered unreliable
    MIN_CCF_CORRELATION = 0.30

    def compute_travel_times(self, dataset: str = 'mekong') -> List[Dict[str, Any]]:
        """
        For each consecutive main-stem pair with sufficient discharge overlap,
        compute the lag that maximises cross-correlation at the native discharge
        resolution. Daily discharge yields daily lags; monthly discharge falls
        back to monthly lags converted to approximate days for reporting.

        Method:
          AR(1) pre-whitening is applied to both series before CCF to remove
          autocorrelation distortion of the lag estimate (Yue & Wang 2002).
          If the peak CCF falls below MIN_CCF_CORRELATION (0.30), the lag is
          flagged as unreliable and lag_days is set to None.

        Returns labelled rows suitable for table display.
        Note: travel times are empirical proxies from discharge CCF, not
        hydraulic routing estimates.
        """
        repo = self._repo_for(dataset)
        valid_stem = [s for s in MEKONG_MAIN_STEM if s in repo.station_index]
        results: List[Dict[str, Any]] = []

        for i in range(len(valid_stem) - 1):
            up_id = valid_stem[i]
            dn_id = valid_stem[i + 1]
            s_up, freq_up = self._discharge_series(repo, up_id)
            s_dn, freq_dn = self._discharge_series(repo, dn_id)
            if s_up is None or s_dn is None or freq_up is None or freq_dn is None:
                continue
            if freq_up != freq_dn:
                continue

            aligned = pd.concat([s_up, s_dn], axis=1, join='inner').dropna()
            aligned.columns = ['up', 'dn']
            is_daily = freq_up == 'daily'
            min_overlap = self.MIN_OVERLAP_DAYS if is_daily else self.MIN_OVERLAP_MONTHS
            max_lag = self.MAX_LAG_DAYS if is_daily else self.MAX_LAG_MONTHS
            overlap_unit = 'days' if is_daily else 'months'
            lag_to_days = 1 if is_daily else 30
            if len(aligned) < min_overlap:
                continue

            # ── AR(1) pre-whitening ────────────────────────────────────────────
            # Fit AR(1) to each series; compute residuals to remove autocorrelation
            # before CCF, reducing spurious lag inflation.
            def _prewhiten(s: pd.Series) -> np.ndarray:
                arr = s.values.astype(float)
                if len(arr) < 4:
                    return arr
                rho = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
                rho = float(np.clip(rho, -0.99, 0.99))
                return arr[1:] - rho * arr[:-1]

            u_raw = _prewhiten(aligned['up'])
            d_raw = _prewhiten(aligned['dn'])

            # Re-normalise pre-whitened residuals
            u_std = float(np.std(u_raw)) or 1.0
            d_std = float(np.std(d_raw)) or 1.0
            u = (u_raw - u_raw.mean()) / u_std
            d = (d_raw - d_raw.mean()) / d_std
            n = len(u)

            best_lag, best_corr = 0, -1.0
            for lag in range(0, max_lag + 1):
                if lag >= n:
                    break
                corr = float(np.dot(u[:n - lag], d[lag:]) / (n - lag))
                if corr > best_corr:
                    best_corr, best_lag = corr, lag

            # Gate: if CCF peak is too weak, lag estimate is unreliable
            reliable = best_corr >= self.MIN_CCF_CORRELATION

            up_meta = repo.station_index[up_id]
            dn_meta = repo.station_index[dn_id]
            results.append({
                'upstream_id':   up_id,
                'downstream_id': dn_id,
                'upstream_name':   up_meta.get('name', up_id) or up_id,
                'downstream_name': dn_meta.get('name', dn_id) or dn_id,
                'lag_days':        best_lag * lag_to_days if reliable else None,
                'correlation':     round(best_corr, 3),
                'overlap_count':   len(aligned),
                'overlap_unit':    overlap_unit,
                'time_resolution': freq_up,
                'lag_reliable':    reliable,
                'lag_note': (
                    f'AR(1) pre-whitened CCF estimate ({overlap_unit} resolution)'
                    if reliable else
                    f'Unreliable — peak CCF ({best_corr:.3f}) below threshold ({self.MIN_CCF_CORRELATION}); lag not reported'
                ),
            })

        return results

    # ── upstream contribution proxy ───────────────────────────────────────────

    def compute_contribution(
        self, target_station: str, dataset: str = 'mekong', include_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Estimate discharge contribution of each upstream station to target.
        Method: discharge-ratio proxy — mean_discharge(upstream) / mean_discharge(target).
        Caveats: does not account for losses, diversions, or parallel tributaries.
        Treat as indicative only.
        """
        repo = self._repo_for(dataset)
        if target_station not in repo.station_index:
            raise ValueError(f'Station {target_station!r} not found.')

        adj = self._build_adjacency(repo)
        ancestors = self._all_upstream(target_station, adj)

        target_fd = repo.station_index[target_station]['feature_details'].get('Discharge')
        target_mean = float(target_fd['mean']) if target_fd else None

        rows: List[Dict[str, Any]] = []
        for anc in sorted(ancestors):
            fd = repo.station_index[anc]['feature_details'].get('Discharge')
            if fd is None or target_mean is None or abs(target_mean) < 1e-10:
                pct = None
            else:
                pct = round(float(fd['mean']) / target_mean * 100, 1)
            meta = repo.station_index[anc]
            rows.append({
                'station':  anc,
                'name':     meta.get('name', anc) or anc,
                'country':  meta.get('country', ''),
                'mean_q':   round(float(fd['mean']), 2) if fd else None,
                'contrib_pct': pct,
            })

        rows.sort(key=lambda r: (r['contrib_pct'] or 0), reverse=True)

        target_meta = repo.station_index[target_station]
        res = {
            'target_station': target_station,
            'target_name':    target_meta.get('name', target_station) or target_station,
            'target_mean_q':  round(target_mean, 2) if target_mean is not None else None,
            'upstream_count': len(ancestors),
            'rows': rows,
            'caveat': (
                'Contribution % is a discharge-ratio proxy: '
                'mean_Q(upstream) / mean_Q(target) × 100. '
                'It does not represent hydraulic routing or account for '
                'losses, evaporation, or tributary splitting.'
            ),
        }
        
        if include_analysis:
            analysis = _generate_contrib_analysis(res)
            if analysis:
                res['analysis'] = analysis
                
        return res

    # ── Plotly network figure ─────────────────────────────────────────────────

    def compute_network_figure(
        self,
        dataset: str = 'mekong',
        travel_times: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        repo = self._repo_for(dataset)
        edges = self._valid_edges(repo)
        idx = repo.station_index
        travel_times = travel_times if travel_times is not None else self.compute_travel_times(dataset)

        main_stem_set = set(MEKONG_MAIN_STEM)
        ordered_stem = [s for s in MEKONG_MAIN_STEM if s in idx]
        main_edges = [(u, d) for u, d in edges if u in main_stem_set and d in main_stem_set]
        trib_edges = [(u, d) for u, d in edges if (u, d) not in main_edges]

        # ── Colour & style constants ──────────────────────────────────────────
        TRIB_CLR = 'rgba(125, 146, 171, 0.22)'
        STEM_EDGE_BASE = 'rgba(49, 91, 120, 0.88)'
        STEM_EDGE_STRONG = '#9adaff'
        STEM_EDGE_MED = '#46bfe9'
        STEM_EDGE_WEAK = 'rgba(95, 132, 160, 0.95)'
        STEM_BORDER = 'rgba(229, 238, 252, 0.92)'
        NODE_HALO = 'rgba(229, 238, 252, 0.16)'
        PANEL_BG = 'rgba(7, 17, 31, 0.78)'
        PANEL_BORDER = 'rgba(157, 176, 209, 0.16)'
        DISCHARGE_SCALE = [
            [0.0, '#12324a'],
            [0.28, '#1d5f74'],
            [0.58, '#2ca3a3'],
            [1.0, '#f3d27b'],
        ]

        def _coord(s: str) -> Tuple[float, float]:
            m = idx[s]
            return float(m['lon']), float(m['lat'])

        all_lons = [float(meta['lon']) for meta in idx.values()]
        all_lats = [float(meta['lat']) for meta in idx.values()]
        lon_min = min(all_lons) - 1.2
        lon_max = max(all_lons) + 1.2
        lat_min = min(all_lats) - 1.0
        lat_max = max(all_lats) + 1.0

        # ── Build in-degree for node sizing ──────────────────────────────────
        in_deg: Dict[str, int] = {s: 0 for s in idx}
        out_deg: Dict[str, int] = {s: 0 for s in idx}
        for _, d in edges:
            if d in in_deg:
                in_deg[d] += 1
        for u, _ in edges:
            if u in out_deg:
                out_deg[u] += 1

        connectivity: Dict[str, int] = {
            s: in_deg.get(s, 0) + out_deg.get(s, 0)
            for s in idx
        }

        travel_lookup = {
            (row['upstream_id'], row['downstream_id']): row
            for row in travel_times
            if row.get('upstream_id') and row.get('downstream_id')
        }

        def _edge_style(u: str, d: str) -> Dict[str, Any]:
            row = travel_lookup.get((u, d))
            corr = row.get('correlation') if row else None
            reliable = bool(row.get('lag_reliable')) if row else False
            if corr is None:
                return {
                    'color': STEM_EDGE_BASE,
                    'width': 3.2,
                    'opacity': 0.78,
                }
            if reliable and corr >= 0.85:
                return {
                    'color': STEM_EDGE_STRONG,
                    'width': 5.2,
                    'opacity': 0.98,
                }
            if reliable and corr >= 0.65:
                return {
                    'color': STEM_EDGE_MED,
                    'width': 4.2,
                    'opacity': 0.94,
                }
            return {
                'color': STEM_EDGE_WEAK,
                'width': 3.6,
                'opacity': 0.84,
            }

        # ── Hover text builder ────────────────────────────────────────────────
        def _hover(s: str) -> str:
            m = idx[s]
            fd_q  = m['feature_details'].get('Discharge')
            fd_wl = m['feature_details'].get('Water_Level')
            name  = (m.get('name', s) or s).replace('_', ' ')
            role  = 'Main stem station' if s in main_stem_set else 'Tributary station'
            lines = [
                f"<b>{name}</b>",
                f"<span style='color:{SOFT}'>{role}</span>",
                f"Country: {m.get('country', '—')}",
                f"Coordinates: {m['lat']:.2f}°N, {m['lon']:.2f}°E",
            ]
            if fd_q:
                lines.append(f"Mean discharge: <b>{fd_q['mean']:.1f} m³/s</b>")
            if fd_wl:
                lines.append(f"Mean water level: {fd_wl['mean']:.2f} m")
            lines.append(f"Incoming links: {in_deg.get(s, 0)}")
            lines.append(f"Outgoing links: {out_deg.get(s, 0)}")
            return '<br>'.join(lines)

        def _edge_hover(u: str, d: str) -> str:
            up_name = (idx[u].get('name', u) or u).replace('_', ' ')
            dn_name = (idx[d].get('name', d) or d).replace('_', ' ')
            row = travel_lookup.get((u, d))
            lines = [
                f"<b>{up_name} → {dn_name}</b>",
                "<span style='color:#9db0d1'>Directed downstream connection</span>",
            ]
            if row:
                lines.append(f"Peak correlation: <b>{row['correlation']:.3f}</b>")
                lag_value = f"{row['lag_days']} day(s)" if row.get('lag_days') is not None else 'Not reported'
                lines.append(f"Inferred lag: {lag_value}")
                lines.append(f"Evidence: {row['lag_note']}")
                lines.append(f"Overlap: {row['overlap_count']} {row.get('overlap_unit', 'steps')}")
            else:
                lines.append("Empirical travel-time evidence unavailable for this link.")
            return '<br>'.join(lines)

        label_specs: Dict[str, Dict[str, Any]] = {
            'Jinghong': {'ax': 34, 'ay': -8, 'callout': True},
            'Chiang_Saen': {'ax': 0, 'ay': -30, 'callout': True},
            'Luang_Prabang': {'ax': 26, 'ay': 14, 'callout': True},
            'Vientiane_KM4': {'ax': 34, 'ay': -20, 'callout': True},
            'Khong_Chiam': {'ax': -46, 'ay': 10, 'callout': True},
            'Stung_Treng': {'ax': 34, 'ay': -6, 'callout': True},
            'Phnom_Penh_Port': {'ax': 40, 'ay': 22, 'callout': True},
            'My_Tho': {'ax': 32, 'ay': 14, 'callout': True},
        }

        fig = go.Figure()

        # ── 1. Tributary edges ─────────────────────────────────────────────────
        trib_lons, trib_lats = [], []
        delta_trib_lons, delta_trib_lats = [], []
        for u, d in trib_edges:
            if u not in idx or d not in idx:
                continue
            ux, uy = _coord(u); dx, dy = _coord(d)
            is_delta_dense = max(ux, dx) > 107.0 or min(uy, dy) < 11.3
            if is_delta_dense:
                delta_trib_lons += [ux, dx, None]
                delta_trib_lats += [uy, dy, None]
            else:
                trib_lons += [ux, dx, None]
                trib_lats += [uy, dy, None]
        if trib_lons:
            fig.add_trace(go.Scatter(
                x=trib_lons, y=trib_lats,
                mode='lines',
                line=dict(color=TRIB_CLR, width=1.05),
                hoverinfo='none',
                showlegend=False,
            ))
        if delta_trib_lons:
            fig.add_trace(go.Scatter(
                x=delta_trib_lons, y=delta_trib_lats,
                mode='lines',
                line=dict(color='rgba(125, 146, 171, 0.12)', width=0.8),
                hoverinfo='none',
                showlegend=False,
            ))

        # ── 2. Main-stem edge underlay for separation ─────────────────────────
        stem_lons, stem_lats = [], []
        for u, d in main_edges:
            if u not in idx or d not in idx:
                continue
            ux, uy = _coord(u); dx, dy = _coord(d)
            stem_lons += [ux, dx, None]
            stem_lats += [uy, dy, None]
        if stem_lons:
            fig.add_trace(go.Scatter(
                x=stem_lons, y=stem_lats,
                mode='lines',
                line=dict(color='rgba(46, 110, 150, 0.34)', width=5.6),
                hoverinfo='none',
                showlegend=False,
            ))

        # ── 3. Main-stem edges split by evidence strength ─────────────────────
        edge_hover_x, edge_hover_y, edge_hover_text = [], [], []
        for u, d in main_edges:
            if u not in idx or d not in idx:
                continue
            style = _edge_style(u, d)
            ux, uy = _coord(u); dx, dy = _coord(d)
            fig.add_trace(go.Scatter(
                x=[ux, dx],
                y=[uy, dy],
                mode='lines',
                line=dict(color=style['color'], width=style['width']),
                opacity=style['opacity'],
                hoverinfo='none',
                showlegend=False,
            ))
            edge_hover_x.append(ux + 0.5 * (dx - ux))
            edge_hover_y.append(uy + 0.5 * (dy - uy))
            edge_hover_text.append(_edge_hover(u, d))

        # ── 4. Invisible main-edge hover points ───────────────────────────────
        if edge_hover_x:
            fig.add_trace(go.Scatter(
                x=edge_hover_x,
                y=edge_hover_y,
                mode='markers',
                marker=dict(size=16, color='rgba(0,0,0,0)'),
                hovertext=edge_hover_text,
                hoverinfo='text',
                name='Connection details',
                showlegend=False,
            ))

        # ── 5. Flow-direction overlays on main-stem edges ─────────────────────
        for u, d in main_edges:
            if u not in idx or d not in idx:
                continue
            ux, uy = _coord(u)
            dx, dy = _coord(d)
            seg_dx = dx - ux
            seg_dy = dy - uy
            seg_len = math.hypot(seg_dx, seg_dy) or 1.0
            tail_frac = 0.48
            head_frac = 0.70 if seg_len > 0.85 else 0.66
            tail_x = ux + tail_frac * seg_dx
            tail_y = uy + tail_frac * seg_dy
            head_x = ux + head_frac * seg_dx
            head_y = uy + head_frac * seg_dy

            # Bright directional shaft sits directly on the connection so flow reads
            # as source -> destination rather than as a detached decorative symbol.
            fig.add_trace(go.Scatter(
                x=[tail_x, head_x],
                y=[tail_y, head_y],
                mode='lines',
                line=dict(color='rgba(239, 248, 255, 0.98)', width=4.4),
                hoverinfo='none',
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[tail_x, head_x],
                y=[tail_y, head_y],
                mode='lines',
                line=dict(color='rgba(70,191,233,0.98)', width=2.4),
                hoverinfo='none',
                showlegend=False,
            ))
            fig.add_annotation(
                x=head_x,
                y=head_y,
                ax=tail_x,
                ay=tail_y,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                text='',
                arrowhead=3,
                arrowsize=1.15,
                arrowwidth=2.6,
                arrowcolor='rgba(239, 248, 255, 0.98)',
                opacity=1.0,
            )
            fig.add_annotation(
                x=head_x,
                y=head_y,
                ax=tail_x,
                ay=tail_y,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                text='',
                arrowhead=3,
                arrowsize=1.05,
                arrowwidth=1.5,
                arrowcolor='rgba(70,191,233,0.98)',
                opacity=1.0,
            )

        # ── 6. Tributary nodes ─────────────────────────────────────────────────
        trib_ids = [s for s in idx if s not in main_stem_set]
        trib_sizes = [7.2 + min(connectivity.get(s, 0), 2) * 1.0 for s in trib_ids]
        fig.add_trace(go.Scatter(
            x=[float(idx[s]['lon']) for s in trib_ids],
            y=[float(idx[s]['lat']) for s in trib_ids],
            mode='markers',
            marker=dict(
                size=trib_sizes,
                color='rgba(161, 177, 198, 0.56)',
                symbol='circle',
                line=dict(color='rgba(211,223,239,0.58)', width=1.0),
                opacity=0.9,
            ),
            hovertext=[_hover(s) for s in trib_ids],
            hoverinfo='text',
            showlegend=False,
        ))

        # ── 7. Main-stem nodes (discharge-coloured, prominent) ─────────────────
        stem_ids = ordered_stem
        stem_q   = [
            float(idx[s]['feature_details']['Discharge']['mean'])
            if idx[s]['feature_details'].get('Discharge') else 0
            for s in stem_ids
        ]
        stem_sizes = [13.5 + min(in_deg.get(s, 0), 2) * 2.2 + min(out_deg.get(s, 0), 2) * 0.8 for s in stem_ids]
        fig.add_trace(go.Scatter(
            x=[float(idx[s]['lon']) for s in stem_ids],
            y=[float(idx[s]['lat']) for s in stem_ids],
            mode='markers',
            marker=dict(
                size=[size + 6 for size in stem_sizes],
                color='rgba(87, 189, 231, 0.11)',
                line=dict(color='rgba(125,211,252,0.16)', width=1),
            ),
            hoverinfo='none',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[float(idx[s]['lon']) for s in stem_ids],
            y=[float(idx[s]['lat']) for s in stem_ids],
            mode='markers',
            marker=dict(
                size=stem_sizes,
                color=stem_q,
                colorscale=DISCHARGE_SCALE,
                colorbar=dict(
                    orientation='h',
                    title=dict(text='Mean discharge (m³/s)', font=dict(size=10, color=SOFT), side='top'),
                    thickness=12,
                    len=0.26,
                    x=0.02,
                    xanchor='left',
                    y=-0.13,
                    yanchor='bottom',
                    tickfont=dict(size=9, color=SOFT),
                    tickformat=',.0f',
                    bgcolor=PANEL_BG,
                    bordercolor=PANEL_BORDER,
                    borderwidth=1,
                ),
                symbol='circle',
                line=dict(color=STEM_BORDER, width=1.8),
                showscale=True,
                cmin=0,
                cmax=max(stem_q, default=1000),
            ),
            hovertext=[_hover(s) for s in stem_ids],
            hoverinfo='text',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[float(idx[s]['lon']) for s in stem_ids],
            y=[float(idx[s]['lat']) for s in stem_ids],
            mode='markers',
            marker=dict(
                size=[max(3.2, size * 0.24) for size in stem_sizes],
                color='rgba(240, 248, 255, 0.72)',
                line=dict(width=0),
            ),
            hoverinfo='none',
            showlegend=False,
        ))

        # ── 8. Selective labels for key stations ──────────────────────────────
        for station, spec in label_specs.items():
            if station not in idx:
                continue
            fig.add_annotation(
                x=float(idx[station]['lon']),
                y=float(idx[station]['lat']),
                xref='x',
                yref='y',
                text=f"<b>{(idx[station].get('name', station) or station).replace('_', ' ')}</b>",
                showarrow=bool(spec.get('callout')),
                arrowhead=0,
                arrowsize=1,
                arrowwidth=1.1,
                arrowcolor='rgba(157, 176, 209, 0.46)',
                ax=spec['ax'],
                ay=spec['ay'],
                axref='pixel',
                ayref='pixel',
                xanchor='center',
                yanchor='middle',
                font=dict(size=10, color=TEXT, family='Inter, sans-serif'),
                bgcolor='rgba(7, 17, 31, 0.9)',
                bordercolor='rgba(157, 176, 209, 0.28)',
                borderwidth=1,
                borderpad=5,
                align='center',
            )

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            xaxis=dict(
                title=dict(text='Longitude (°E)', font=dict(size=11, color=SOFT)),
                range=[lon_min, lon_max],
                gridcolor=GRID_LIGHT,
                zeroline=False,
                tickfont=dict(size=10, color=SOFT),
                linecolor='rgba(148,163,184,0.12)',
                dtick=2,
                automargin=True,
                showspikes=False,
            ),
            yaxis=dict(
                title=dict(text='Latitude (°N)', font=dict(size=11, color=SOFT)),
                range=[lat_min, lat_max],
                gridcolor=GRID_LIGHT,
                zeroline=False,
                tickfont=dict(size=10, color=SOFT),
                linecolor='rgba(148,163,184,0.12)',
                dtick=2,
                automargin=True,
                showspikes=False,
            ),
            showlegend=False,
            margin=dict(l=76, r=30, t=18, b=88),
            height=760,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor='rgba(7,17,31,0.92)',
                bordercolor='rgba(125,211,252,0.34)',
                font=dict(size=12, color=TEXT, family='Inter, sans-serif'),
            ),
        )

        fig.update_xaxes(showgrid=True, gridcolor=GRID_LIGHT, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_STRONG, zeroline=False)

        return plotly.io.to_json(fig)

    # ── full analysis (called by API) ─────────────────────────────────────────

    def compute_full_network(self, dataset: str = 'mekong', include_analysis: bool = False) -> Dict[str, Any]:
        repo = self._repo_for(dataset)
        adj = self._build_adjacency(repo)
        edges = self._valid_edges(repo)

        # Node list with adjacency info
        nodes = []
        for s, meta in repo.station_index.items():
            fd_q = meta['feature_details'].get('Discharge')
            nodes.append({
                'id':         s,
                'name':       (meta.get('name', s) or s).replace('_', ' '),
                'country':    meta.get('country', ''),
                'lat':        meta['lat'],
                'lon':        meta['lon'],
                'main_stem':  s in set(MEKONG_MAIN_STEM),
                'upstream':   adj[s]['upstream'],
                'downstream': adj[s]['downstream'],
                'mean_q':     round(float(fd_q['mean']), 2) if fd_q else None,
                'has_discharge': fd_q is not None,
            })

        edge_list = [
            {'upstream': u, 'downstream': d,
             'main_stem': u in set(MEKONG_MAIN_STEM) and d in set(MEKONG_MAIN_STEM)}
            for u, d in edges
        ]

        travel_times = self.compute_travel_times(dataset)

        result = {
            'dataset':      dataset,
            'nodes':        nodes,
            'edges':        edge_list,
            'main_stem':    [s for s in MEKONG_MAIN_STEM if s in repo.station_index],
            'node_count':   len(nodes),
            'edge_count':   len(edge_list),
            'figure':       self.compute_network_figure(dataset, travel_times=travel_times),
            'travel_times': travel_times,
            'note': (
                'Main-stem links are weighted by empirical discharge travel-time evidence; tributaries provide documented network context.'
            ),
            'graph_guide': (
                'Fill shows mean discharge. Larger main-stem markers indicate more connected junctions. '
                'Brighter, thicker path segments indicate stronger empirical support. Arrows indicate downstream direction.'
            ),
            'travel_time_note': (
                'Travel times are estimated from the peak lag of AR(1) pre-whitened discharge '
                'cross-correlation series at native time resolution, reported in days. This is an empirical proxy — not a hydraulic routing result. '
                f'Lags with CCF peak < {self.MIN_CCF_CORRELATION} are flagged as unreliable and reported as null.'
            ),
            'contribution_note': (
                'Station contribution estimates (where shown) are discharge-ratio proxies: '
                'mean_Q(upstream) / mean_Q(target) × 100. '
                'They do not account for losses, abstractions, tributary inflows, or routing delays. '
                'Treat as exploratory basin connectivity indicators, not hydraulic routing outputs.'
            ),
            'analysis_strength': 'exploratory_empirical',
        }

        if include_analysis:
            analysis = _generate_network_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
