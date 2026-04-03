import unittest
from unittest.mock import patch

from services.network_service import _generate_network_analysis


class NetworkAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            'dataset': 'mekong',
            'node_count': 42,
            'edge_count': 51,
            'main_stem': ['Jinghong', 'Chiang_Saen', 'Pakse', 'My_Tho'],
            'note': 'Topology is based on published Mekong River Commission basin maps.',
            'travel_times': [
                {
                    'upstream_name': 'Chiang_Saen',
                    'downstream_name': 'Chiang_Khan',
                    'lag_months': 1,
                    'correlation': 0.82,
                    'overlap_months': 96,
                },
                {
                    'upstream_name': 'Pakse',
                    'downstream_name': 'Stung_Treng',
                    'lag_months': 2,
                    'correlation': 0.71,
                    'overlap_months': 88,
                },
            ],
        }

    def test_network_analysis_returns_markdown_html(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}, clear=False):
            with patch('services.network_service._gemini_generate', return_value='## Network Summary\nText\n\n## Connectivity Structure\n- A\n- B\n- C\n- D\n\n## Travel-Time Interpretation\n- A\n- B\n- C\n- D\n\n## Operational Relevance\n- A\n- B\n- C'):
                analysis = _generate_network_analysis(self.result)
        self.assertIn('<h2>', analysis)
        self.assertIn('Network Summary', analysis)
        self.assertIn('Connectivity Structure', analysis)

    def test_network_analysis_falls_back_when_ai_fails(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}, clear=False):
            with patch('services.network_service._gemini_generate', side_effect=Exception('429 RESOURCE_EXHAUSTED')):
                analysis = _generate_network_analysis(self.result)
        self.assertIn('Network Summary', analysis)
        self.assertIn('Live AI analysis unavailable', analysis)
        self.assertIn('Topology extent', analysis)


if __name__ == '__main__':
    unittest.main()
