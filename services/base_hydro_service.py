"""
base_hydro_service.py
~~~~~~~~~~~~~~~~~~~~~
Base class for HydroVision analysis services.

Provides the two _find_repo patterns and the _load_series helpers that are
otherwise copy-pasted verbatim across every service.  New services should
inherit from BaseHydroService instead of duplicating this boilerplate.

Existing services do NOT need to be migrated immediately — they will continue
to work as-is.  Migrate incrementally when a service is otherwise being edited.

Usage::

    from .base_hydro_service import BaseHydroService

    class MyService(BaseHydroService):
        def compute(self, dataset: str, station: str, feature: str):
            repo = self._find_repo_by_dataset(dataset)   # dataset-keyed
            # or
            repo = self._find_repo_by_station(station)   # station-keyed
            ts   = self._load_series(repo, station, feature)      # soft → None
            ts   = self._load_series_strict(repo, station, feature)  # hard → raises
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .data_loader import SeriesRequest


class BaseHydroService:
    """
    Abstract base providing shared repository access helpers.

    Subclasses must call ``super().__init__(repository)`` or assign
    ``self.repo`` themselves before calling any helper.
    """

    def __init__(self, repository) -> None:
        self.repo = repository

    # ── Repository lookup ─────────────────────────────────────────────────────

    def _find_repo_by_dataset(self, dataset: str):
        """
        Return the DataRepository whose ``dataset`` attribute matches *dataset*.

        Works with both a single repository and a composite repository that
        exposes a ``repos`` list.  Returns ``None`` when no match is found.

        Used by services that receive a *dataset* parameter (animation,
        decomposition, climate, wavelet, changepoint, …).
        """
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if r.dataset == dataset), None)
        if getattr(self.repo, 'dataset', '') == dataset:
            return self.repo
        return None

    def _find_repo_by_station(self, station: str):
        """
        Return the DataRepository whose ``station_index`` contains *station*.

        Used by services that receive a *station* parameter directly (extreme,
        scenario, quality, …).  Returns ``None`` when no match is found.
        """
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if station in r.station_index), None)
        if station in getattr(self.repo, 'station_index', {}):
            return self.repo
        return None

    # ── Series loading ────────────────────────────────────────────────────────

    def _load_series(
        self,
        repo,
        station: str,
        feature: str,
    ) -> Optional[pd.Series]:
        """
        Load the full historical time series for *station* / *feature*.

        Returns a ``pd.Series`` indexed by ``pd.DatetimeIndex`` with NaNs
        dropped, or ``None`` on any error (missing station, missing feature,
        data-loader exception).

        Use this variant when the caller can tolerate a missing series and
        will skip the station gracefully (e.g. map-building loops).
        """
        try:
            fd = repo.station_index[station]['feature_details'][feature]
            req = SeriesRequest(
                station=station, feature=feature,
                start_date=fd['start_date'], end_date=fd['end_date'],
            )
            df = repo.get_feature_series(req)
            ts = df.set_index('Timestamp')['Value'].sort_index()
            ts.index = pd.to_datetime(ts.index)
            return ts.dropna()
        except Exception:
            return None

    def _load_series_strict(
        self,
        repo,
        station: str,
        feature: str,
    ) -> pd.Series:
        """
        Load the full historical time series for *station* / *feature*.

        Unlike ``_load_series``, this variant lets exceptions propagate so
        the caller receives a clear error message.  NaNs are still dropped.

        Use this variant in entry-point methods where a missing series should
        abort the computation entirely (e.g. single-station analyses).
        """
        fd = repo.station_index[station]['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = repo.get_feature_series(req)
        ts = df.set_index('Timestamp')['Value'].sort_index()
        ts.index = pd.to_datetime(ts.index)
        return ts.dropna()
