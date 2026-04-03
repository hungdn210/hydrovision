import unittest
from unittest.mock import patch

from services.quality_service import generate_quality_analysis


class TestQualityAnalysis(unittest.TestCase):
    def test_completeness_fallback_returns_html(self):
        result = {
            'station': 'Ban_Chot',
            'feature': 'Discharge',
            'overall_pct': 82.4,
            'missing_months': 3,
            'low_months': 5,
            'total_months': 48,
        }
        with patch.dict('os.environ', {}, clear=True):
            analysis = generate_quality_analysis('completeness', result)
        self.assertIn('<h2>', analysis)
        self.assertIn('Quality Summary', analysis)
        self.assertIn('Coverage', analysis)


if __name__ == '__main__':
    unittest.main()
