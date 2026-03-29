"""
network_service.py
~~~~~~~~~~~~~~~~~~
Spatial Network Analysis for the Mekong basin.

Topology: documented Mekong/Lower Mekong Basin river system — main-stem order
and tributary connections are hardcoded and clearly labelled.  Sub-basin GeoJSON
is only a single boundary polygon, so connections are derived from published
basin reports and station coordinates rather than automated spatial joins.

Travel times: empirical cross-correlation of monthly discharge series.
Contribution: discharge-ratio proxy (labelled as estimate, not hydraulic routing).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io

from .data_loader import DataRepository, MultiDataRepository, SeriesRequest


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

    MAX_LAG_MONTHS = 6     # max cross-corr lag (monthly series)
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

    def _monthly_discharge(
        self, repo: DataRepository, station: str
    ) -> Optional[pd.Series]:
        fd = repo.station_index[station]['feature_details'].get('Discharge')
        if fd is None:
            return None
        try:
            req = SeriesRequest(
                station=station, feature='Discharge',
                start_date=fd['start_date'], end_date=fd['end_date'],
            )
            df = repo.get_feature_series(req)
            return (
                df.set_index('Timestamp')['Value']
                .resample('MS').mean()
                .dropna()
            )
        except Exception:
            return None

    # ── travel time via cross-correlation ────────────────────────────────────

    def compute_travel_times(self, dataset: str = 'mekong') -> List[Dict[str, Any]]:
        """
        For each consecutive main-stem pair with sufficient discharge overlap,
        compute the monthly lag that maximises cross-correlation.
        Returns labelled rows suitable for table display.
        """
        repo = self._repo_for(dataset)
        valid_stem = [s for s in MEKONG_MAIN_STEM if s in repo.station_index]
        results: List[Dict[str, Any]] = []

        for i in range(len(valid_stem) - 1):
            up_id = valid_stem[i]
            dn_id = valid_stem[i + 1]
            s_up = self._monthly_discharge(repo, up_id)
            s_dn = self._monthly_discharge(repo, dn_id)
            if s_up is None or s_dn is None:
                continue

            aligned = pd.concat([s_up, s_dn], axis=1, join='inner').dropna()
            aligned.columns = ['up', 'dn']
            if len(aligned) < self.MIN_OVERLAP_MONTHS:
                continue

            # Normalise
            u = (aligned['up'] - aligned['up'].mean()) / (aligned['up'].std() + 1e-12)
            d = (aligned['dn'] - aligned['dn'].mean()) / (aligned['dn'].std() + 1e-12)
            n = len(u)

            best_lag, best_corr = 0, -1.0
            for lag in range(0, self.MAX_LAG_MONTHS + 1):
                if lag >= n:
                    break
                corr = float(np.dot(u.values[:n - lag], d.values[lag:]) / (n - lag))
                if corr > best_corr:
                    best_corr, best_lag = corr, lag

            up_meta = repo.station_index[up_id]
            dn_meta = repo.station_index[dn_id]
            results.append({
                'upstream_id':   up_id,
                'downstream_id': dn_id,
                'upstream_name':   up_meta.get('name', up_id) or up_id,
                'downstream_name': dn_meta.get('name', dn_id) or dn_id,
                'lag_months':      best_lag,
                'correlation':     round(best_corr, 3),
                'overlap_months':  len(aligned),
            })

        return results

    # ── upstream contribution proxy ───────────────────────────────────────────

    def compute_contribution(
        self, target_station: str, dataset: str = 'mekong'
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
        return {
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

    # ── Plotly network figure ─────────────────────────────────────────────────

    def compute_network_figure(self, dataset: str = 'mekong') -> str:
        repo = self._repo_for(dataset)
        edges = self._valid_edges(repo)
        idx = repo.station_index

        main_stem_set = set(MEKONG_MAIN_STEM)

        # Classify edges
        main_edges   = [(u, d) for u, d in edges if u in main_stem_set and d in main_stem_set]
        trib_edges   = [(u, d) for u, d in edges if (u, d) not in main_edges]

        def _coord(s: str) -> Tuple[float, float]:
            m = idx[s]
            return float(m['lon']), float(m['lat'])

        def _edge_trace(
            edge_list: List[Tuple[str, str]],
            color: str,
            width: float,
            name: str,
        ) -> go.Scatter:
            lons, lats = [], []
            for u, d in edge_list:
                if u not in idx or d not in idx:
                    continue
                ux, uy = _coord(u)
                dx, dy = _coord(d)
                lons += [ux, dx, None]
                lats += [uy, dy, None]
            return go.Scatter(
                x=lons, y=lats,
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='none',
                name=name,
                showlegend=True,
            )

        # Build in-degree for node sizing
        in_deg: Dict[str, int] = {s: 0 for s in idx}
        for _, d in edges:
            if d in in_deg:
                in_deg[d] += 1

        # Node attributes
        station_ids = list(idx.keys())
        lons_n = [float(idx[s]['lon']) for s in station_ids]
        lats_n = [float(idx[s]['lat']) for s in station_ids]

        q_means = []
        for s in station_ids:
            fd = idx[s]['feature_details'].get('Discharge')
            q_means.append(float(fd['mean']) if fd else None)

        sizes = [10 + in_deg.get(s, 0) * 3 for s in station_ids]
        colors = [q if q is not None else 0 for q in q_means]
        text_labels = [
            (idx[s].get('name', s) or s).replace('_', ' ')
            for s in station_ids
        ]
        hover_texts = []
        for s in station_ids:
            m = idx[s]
            fd_q = m['feature_details'].get('Discharge')
            fd_wl = m['feature_details'].get('Water_Level')
            parts = [
                f"<b>{(m.get('name', s) or s).replace('_', ' ')}</b>",
                f"Country: {m.get('country', '—')}",
                f"Lat/Lon: {m['lat']:.3f}°N, {m['lon']:.3f}°E",
            ]
            if fd_q:
                parts.append(f"Mean discharge: {fd_q['mean']:.1f} m³/s")
            if fd_wl:
                parts.append(f"Mean water level: {fd_wl['mean']:.2f} m")
            stem_tag = ' <i>(main stem)</i>' if s in main_stem_set else ''
            parts.append(f"Role: {'Main stem' + stem_tag if s in main_stem_set else 'Tributary'}")
            hover_texts.append('<br>'.join(parts))

        symbol_list = [
            'diamond' if s in main_stem_set else 'circle'
            for s in station_ids
        ]

        fig = go.Figure()

        # Tributary edges first (behind)
        fig.add_trace(_edge_trace(
            trib_edges,
            color='rgba(148,163,184,0.55)', width=1.4,
            name='Tributary channel',
        ))
        # Main stem edges on top
        fig.add_trace(_edge_trace(
            main_edges,
            color='rgba(56,189,248,0.85)', width=3.0,
            name='Main Mekong stem',
        ))

        # Nodes
        fig.add_trace(go.Scatter(
            x=lons_n, y=lats_n,
            mode='markers+text',
            text=text_labels,
            textposition='top center',
            textfont=dict(size=8, color='rgba(148,163,184,0.9)'),
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Blues',
                colorbar=dict(
                    title='Mean discharge (m³/s)',
                    thickness=12,
                    len=0.6,
                    x=1.02,
                ),
                symbol=symbol_list,
                line=dict(color='rgba(255,255,255,0.7)', width=1),
                showscale=True,
                cmin=0,
                cmax=max((c for c in colors if c), default=1000),
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            name='Gauging stations',
        ))

        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text='Mekong River Network — Spatial Station Topology',
                x=0.5, xanchor='center',
                font=dict(size=16),
            ),
            xaxis=dict(
                title='Longitude (°E)',
                range=[97, 110],
                gridcolor='rgba(255,255,255,0.06)',
                zeroline=False,
            ),
            yaxis=dict(
                title='Latitude (°N)',
                range=[7, 24],
                gridcolor='rgba(255,255,255,0.06)',
                zeroline=False,
                scaleanchor='x',
                scaleratio=1,
            ),
            paper_bgcolor='rgba(7,17,31,0)',
            plot_bgcolor='rgba(7,17,31,0)',
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.12,
                xanchor='center', x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=50, r=80, t=60, b=60),
            font=dict(color='#e5eefc'),
            height=620,
        )

        return plotly.io.to_json(fig)

    # ── full analysis (called by API) ─────────────────────────────────────────

    def compute_full_network(self, dataset: str = 'mekong') -> Dict[str, Any]:
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

        return {
            'dataset':      dataset,
            'nodes':        nodes,
            'edges':        edge_list,
            'main_stem':    [s for s in MEKONG_MAIN_STEM if s in repo.station_index],
            'node_count':   len(nodes),
            'edge_count':   len(edge_list),
            'figure':       self.compute_network_figure(dataset),
            'travel_times': self.compute_travel_times(dataset),
            'note': (
                'Topology is based on published Mekong River Commission basin maps. '
                'Tributary assignments are approximate. '
                'Travel times are empirical (monthly discharge cross-correlation). '
            ),
        }
