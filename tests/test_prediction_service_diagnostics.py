import unittest
from unittest.mock import MagicMock

import pandas as pd

from services.prediction_service import PredictionService


class TestPredictionDiagnostics(unittest.TestCase):
    def test_diagnostics_use_trained_model_fit_when_available(self):
        repo = MagicMock()
        repo.feature_units = {'Discharge': 'm^3/s'}
        service = PredictionService(repo)

        frame = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=20, freq='D'),
            'Value': [100.0] * 20,
        })
        hist_fit_df = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=20, freq='D'),
            'ModelFit': [90.0] * 20,
        })

        result = service._build_diagnostics_figure(
            frame,
            feature='Discharge',
            frequency='daily',
            hist_fit_df=hist_fit_df,
            source_label='FlowNet',
        )

        self.assertIsNotNone(result)
        self.assertIn('FlowNet historical fit', result['summary'])
        self.assertIn('Standard deviation is 0.000', result['summary'])


if __name__ == '__main__':
    unittest.main()
