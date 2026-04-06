import unittest
from unittest.mock import MagicMock

import pandas as pd

from services.risk_service import RiskService


class RiskServiceTests(unittest.TestCase):
    def test_compute_risk_map_seasonal_mode_uses_calendar_month_without_index_mode_error(self):
        repo = MagicMock()
        repo.dataset = 'mekong'
        repo.feature_units = {'Discharge': 'm3/s'}
        repo.station_index = {
            'Station_A': {
                'lat': 15.2,
                'lon': 105.1,
                'features': ['Discharge'],
                'feature_details': {
                    'Discharge': {
                        'start_date': '2020-01-01',
                        'end_date': '2021-12-31',
                    }
                },
                'name': 'Station A',
            }
        }

        multi_repo = MagicMock()
        multi_repo.repos = [repo]

        service = RiskService(multi_repo)

        ts = pd.Series(
            [
                10, 12, 11, 13, 14, 15, 12, 11, 10, 9, 8, 7,
                11, 12, 13, 14, 15, 16, 13, 12, 11, 10, 9, 8,
                12, 13, 14, 15, 16, 17, 14, 13, 12, 11, 10, 9,
            ],
            index=pd.date_range('2020-01-01', periods=36, freq='MS'),
        )
        service._load_series = MagicMock(return_value=ts)

        result = service.compute_risk_map(
            dataset='mekong',
            feature='Discharge',
            lookback=7,
            include_analysis=False,
            seasonal=True,
        )

        self.assertEqual(result['dataset'], 'mekong')
        self.assertEqual(result['feature'], 'Discharge')
        self.assertEqual(result['n_stations'], 1)
        self.assertEqual(len(result['stations']), 1)
        self.assertEqual(result['percentile_mode'], 'seasonal')
        self.assertTrue(result['stations'][0]['percentile_mode'].startswith('seasonal_month_'))


if __name__ == '__main__':
    unittest.main()
