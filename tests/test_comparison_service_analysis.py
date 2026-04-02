import unittest
from unittest.mock import patch

from services.comparison_service import _generate_component_analysis


class TestComparisonAnalysis(unittest.TestCase):
    def setUp(self):
        self.correlation = {
            'dataset': 'mekong',
            'feature': 'Discharge',
            'n_stations': 3,
            'total_available': 3,
            'capped': False,
            'stations': ['Ban Chot', 'Chiang Saen', 'Pakse'],
            'mean_correlations': [0.81, 0.75, 0.72],
            'matrix': [
                [1.0, 0.88, 0.70],
                [0.88, 1.0, 0.66],
                [0.70, 0.66, 1.0],
            ],
        }
        self.leaderboard = {
            'year': 2003,
            'above_normal': 1,
            'below_normal': 1,
            'total_stations': 2,
            'rows': [
                {'name': 'Ban Chot', 'anomaly_pct': 42.5, 'direction': 'above', 'year_mean': 123.4, 'clim_mean': 86.6, 'unit': 'm^3/s', 'level': 'warning'},
                {'name': 'Pakse', 'anomaly_pct': -31.2, 'direction': 'below', 'year_mean': 210.0, 'clim_mean': 305.2, 'unit': 'm^3/s', 'level': 'watch'},
            ],
        }
        self.summary = {
            'dataset': 'mekong',
            'feature': 'Discharge',
            'unit': 'm^3/s',
            'active_stations': 3,
            'total_stations': 3,
            'basin_mean': 154.2,
            'basin_median': 140.0,
            'basin_std': 41.5,
            'basin_min': 86.6,
            'basin_max': 236.0,
            'p10': 95.0,
            'p25': 110.0,
            'p75': 190.0,
            'p90': 220.0,
            'spatial_cv_pct': 26.9,
            'highest_station': {'name': 'Pakse', 'mean': 236.0},
            'lowest_station': {'name': 'Ban Chot', 'mean': 86.6},
            'trends_computed': True,
            'trends': {'rising': 1, 'stable': 1, 'falling': 1},
            'avg_imputation_pct': 1.8,
            'total_observations': 999,
        }

    def test_returns_html_fallback_for_correlation_when_ai_disabled(self):
        with patch.dict('os.environ', {}, clear=True):
            analysis = _generate_component_analysis('correlation', self.correlation, 'Discharge')
        self.assertIn('<h2>', analysis)
        self.assertIn('Matrix Overview', analysis)
        self.assertIn('Ban Chot', analysis)

    def test_returns_html_fallback_for_leaderboard_when_ai_disabled(self):
        with patch.dict('os.environ', {}, clear=True):
            analysis = _generate_component_analysis('leaderboard', self.leaderboard, 'Discharge')
        self.assertIn('<h2>', analysis)
        self.assertIn('Year Context', analysis)
        self.assertIn('Pakse', analysis)

    def test_returns_html_fallback_for_summary_when_ai_disabled(self):
        with patch.dict('os.environ', {}, clear=True):
            analysis = _generate_component_analysis('summary', self.summary, 'Discharge')
        self.assertIn('<h2>', analysis)
        self.assertIn('Basin Snapshot', analysis)
        self.assertIn('Pakse', analysis)


if __name__ == '__main__':
    unittest.main()
