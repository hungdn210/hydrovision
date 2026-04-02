import unittest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import services.analysis_service as analysis_service
from services.analysis_service import AnalysisService

class TestAnalysisService(unittest.TestCase):
    def test_analyse_free_multi_per_graph_stats(self):
        # Mock repository and chart service
        mock_repo = MagicMock()
        
        # Frame 1: Station A
        frame1 = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=10, freq='D'),
            'Value': [1,2,3,4,5,6,7,8,9,10],
            'Station': ['Station_A'] * 10,
            'Feature': ['Water_Level'] * 10,
            'Unit': ['m'] * 10,
            'Imputed': ['No'] * 10
        })

        # Frame 2: Station B
        frame2 = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=10, freq='D'),
            'Value': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'Station': ['Station_B'] * 10,
            'Feature': ['Water_Level'] * 10,
            'Unit': ['m'] * 10,
            'Imputed': ['No'] * 10
        })

        mock_repo.get_feature_series.side_effect = [frame1, frame2, frame1, frame2]
        
        # Mock repo dataset benchmark logic
        mock_repo_attr = MagicMock()
        mock_repo_attr.dataset = 'test_dataset'
        mock_repo_attr.station_index = {
            'Station_A': {'feature_details': {'Water_Level': {'mean': 5.5}}},
            'Station_B': {'feature_details': {'Water_Level': {'mean': 550.0}}}
        }
        mock_repo._station_to_repo = {'Station_A': mock_repo_attr, 'Station_B': mock_repo_attr}

        mock_chart = MagicMock()
        
        # Let pick_three_graphs return multiple mocked graphs configs
        mock_chart.generate_chart.side_effect = [
            {'graph_type': 'GraphType1', 'series': [{'station': 'Station_A', 'feature': 'Water_Level', 'start_date': '2020-01-01', 'end_date': '2020-01-10'}]},
            {'graph_type': 'GraphType2', 'series': [{'station': 'Station_B', 'feature': 'Water_Level', 'start_date': '2020-01-01', 'end_date': '2020-01-10'}]}
        ]

        service = AnalysisService(mock_repo, mock_chart)
        
        payload = {
            'series': [{'station': 'Station_A', 'feature': 'Water_Level'}]
        }
        
        # We also need to mock _pick_three_graphs to control the exact configs
        with patch.object(service, '_pick_three_graphs') as mock_pick:
            mock_pick.return_value = [
                {'graph_type': 'T1', 'series': [{'station': 'Station_A', 'feature': 'Water_Level'}], 'label': 'L1', 'focus': 'F1'},
                {'graph_type': 'T2', 'series': [{'station': 'Station_B', 'feature': 'Water_Level'}], 'label': 'L2', 'focus': 'F2'}
            ]
            
            result = service.analyse_free_multi(payload)
            
            # Check there are two graph blocks in the result
            self.assertEqual(len(result['graphs']), 2)
            
            # Graph 1 (Station_A) findings should have mean around 5.5
            g1_findings = result['graphs'][0]['analysis']['findings']
            self.assertEqual(len(g1_findings), 1)
            self.assertIn('5.500', g1_findings[0]['mean'])
            
            # Graph 2 (Station_B) findings should have mean around 550
            g2_findings = result['graphs'][1]['analysis']['findings']
            self.assertEqual(len(g2_findings), 1)
            self.assertIn('550.000', g2_findings[0]['mean'])

            # Benchmark should include both unique stations
            benchmarks = result['benchmark']
            self.assertEqual(len(benchmarks), 2)
            stations_in_benchmarks = {b['station'] for b in benchmarks}
            self.assertEqual({'Station_A', 'Station_B'}, stations_in_benchmarks)

    def test_analyse_free_multi_falls_back_when_ai_generation_fails(self):
        mock_repo = MagicMock()
        frame = pd.DataFrame({
            'Timestamp': pd.date_range('2020-01-01', periods=10, freq='D'),
            'Value': [1,2,3,4,5,6,7,8,9,10],
            'Station': ['Station_A'] * 10,
            'Feature': ['Water_Level'] * 10,
            'Unit': ['m'] * 10,
            'Imputed': ['No'] * 10
        })
        mock_repo.get_feature_series.return_value = frame

        repo_meta = MagicMock()
        repo_meta.dataset = 'test_dataset'
        repo_meta.station_index = {
            'Station_A': {'feature_details': {'Water_Level': {'mean': 5.5}}},
            'Station_B': {'feature_details': {'Water_Level': {'mean': 7.0}}},
        }
        mock_repo._station_to_repo = {'Station_A': repo_meta}

        mock_chart = MagicMock()
        mock_chart.generate_chart.return_value = {
            'graph_type': 'GraphType1',
            'series': [{'station': 'Station_A', 'feature': 'Water_Level', 'start_date': '2020-01-01', 'end_date': '2020-01-10'}]
        }

        service = AnalysisService(mock_repo, mock_chart)
        payload = {
            'series': [{'station': 'Station_A', 'feature': 'Water_Level'}]
        }

        with patch.object(service, '_pick_three_graphs') as mock_pick, \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'demo-key'}), \
             patch('services.analysis_service._gemini_generate', side_effect=Exception('429 RESOURCE_EXHAUSTED')):
            mock_pick.return_value = [
                {'graph_type': 'T1', 'series': [{'station': 'Station_A', 'feature': 'Water_Level'}], 'label': 'L1', 'focus': 'F1'}
            ]

            result = service.analyse_free_multi(payload)

            self.assertEqual(len(result['graphs']), 1)
            summary = result['graphs'][0]['analysis']['summary']
            self.assertIn('AI status', summary)
            self.assertIn('Primary series', summary)

    def test_ai_cache_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'ai_cache.json'
            with patch.object(analysis_service, '_CACHE_PATH', cache_path):
                analysis_service._AI_RESPONSE_CACHE.clear()
                analysis_service._save_ai_cache({'abc': 'cached response'})
                loaded = analysis_service._load_ai_cache()
                self.assertEqual({'abc': 'cached response'}, loaded)
                self.assertTrue(cache_path.exists())

if __name__ == '__main__':
    unittest.main()
