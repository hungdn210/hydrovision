import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from services.scenario_service import ScenarioService


class TestScenarioService(unittest.TestCase):
    def setUp(self):
        self.service = ScenarioService(repository=None)

    def test_compute_sensitivity_detects_lagged_driver_response(self):
        idx = pd.date_range('2018-01-01', periods=48, freq='MS')
        seasonal_driver = np.array([40 + (i % 12) * 1.5 for i in range(48)], dtype=float)
        seasonal_target = np.array([100 + (i % 12) * 2.0 for i in range(48)], dtype=float)

        shock = np.zeros(48, dtype=float)
        for pos, val in [(3, 18), (10, 12), (16, -10), (23, 15), (31, -8), (38, 14), (44, -11)]:
            shock[pos] = val

        driver = pd.Series(seasonal_driver + shock, index=idx)
        target = pd.Series(seasonal_target + np.roll(shock, 1) * 0.7, index=idx)
        target.iloc[0] = seasonal_target[0]

        sensitivity = self.service._compute_sensitivity(target, driver, direct=False)

        self.assertEqual('distributed_lag_anomaly_response', sensitivity['model_type'])
        self.assertEqual(1, sensitivity['dominant_lag'])
        self.assertGreater(sensitivity['r_value'], 0.3)

    def test_simulate_scenario_response_propagates_month_varying_impacts(self):
        baseline = pd.Series([10.0] * 8, index=pd.date_range('2024-01-01', periods=8, freq='MS'))
        sensitivity = {
            'direct': False,
            'target_persistence': 0.4,
            'driver_lag_coeffs': [0.2, 0.3, 0.1, 0.0],
        }

        scenario, delta, delta_pct = self.service._simulate_scenario_response(
            baseline=baseline,
            scale_pct=20,
            duration_months=2,
            start_offset=0,
            sensitivity=sensitivity,
        )

        self.assertEqual(len(scenario), 8)
        self.assertAlmostEqual(delta_pct.iloc[0], 4.0, places=3)
        self.assertNotAlmostEqual(delta_pct.iloc[1], delta_pct.iloc[0], places=3)
        self.assertGreater(delta.iloc[2], 0.0)

    def test_run_scenario_rejects_weak_relationship(self):
        repo = MagicMock()
        repo.station_index = {
            'Ban_Chot': {
                'features': ['Discharge', 'Rainfall'],
                'feature_details': {
                    'Discharge': {'start_date': '2020-01-01', 'end_date': '2023-12-31'},
                    'Rainfall': {'start_date': '2020-01-01', 'end_date': '2023-12-31'},
                },
            }
        }
        repo.feature_units = {'Discharge': 'm^3/s'}
        repo.feature_frequency = {'Discharge': 'daily'}
        service = ScenarioService(repository=repo)

        idx = pd.date_range('2020-01-01', periods=48, freq='MS')
        discharge = pd.Series(np.linspace(10, 20, 48), index=idx)
        rainfall = pd.Series(np.linspace(100, 110, 48), index=idx)

        with patch.object(service, '_find_repo', return_value=repo), \
             patch.object(service, '_load_series', side_effect=[discharge, rainfall]), \
             patch.object(service, '_compute_sensitivity', return_value={
                 'direct': False,
                 'model_type': 'distributed_lag_anomaly_response',
                 'n_months': 48,
                 'r_value': 0.10,
                 'fit_r2': 0.03,
                 'cumulative_response': 0.01,
             }):
            with self.assertRaisesRegex(ValueError, 'Relationship too weak for a reliable scenario estimate.'):
                service.run_scenario(
                    station='Ban_Chot',
                    target_feature='Discharge',
                    driver_feature='Rainfall',
                    scale_pct=20,
                    duration_months=3,
                    start_offset=0,
                    model='FlowNet',
                )


if __name__ == '__main__':
    unittest.main()
