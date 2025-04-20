import unittest
import numpy as np
import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import WarehouseAnalytics

class TestIdleTimeCalculation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.analytics = WarehouseAnalytics()
        
        # Sample tracking results with employee tracks
        # Format [track_id, class_id, x1, y1, x2, y2, frame_idx]
        self.sample_tracks = {
            'tracks': [
                # Employee 1 - mostly moving
                [1, 0, 100, 100, 120, 200, 1],  # frame 1
                [1, 0, 105, 110, 125, 210, 2],  # frame 2
                [1, 0, 110, 120, 130, 220, 3],  # frame 3
                [1, 0, 115, 130, 135, 230, 4],  # frame 4
                [1, 0, 120, 140, 140, 240, 5],  # frame 5
                [1, 0, 120, 141, 140, 241, 6],  # frame 6 - idle
                [1, 0, 121, 142, 141, 242, 7],  # frame 7 - idle
                [1, 0, 120, 142, 140, 242, 8],  # frame 8 - idle
                [1, 0, 130, 150, 150, 250, 9],  # frame 9
                [1, 0, 140, 160, 160, 260, 10], # frame 10
                
                # Employee 2 - mostly idle
                [2, 0, 200, 200, 220, 300, 1],  # frame 1
                [2, 0, 201, 200, 221, 300, 2],  # frame 2 - idle
                [2, 0, 201, 201, 221, 301, 3],  # frame 3 - idle
                [2, 0, 200, 201, 220, 301, 4],  # frame 4 - idle
                [2, 0, 201, 200, 221, 300, 5],  # frame 5 - idle
                [2, 0, 220, 220, 240, 320, 6],  # frame 6
                [2, 0, 221, 221, 241, 321, 7],  # frame 7 - idle
                [2, 0, 222, 222, 242, 322, 8],  # frame 8 - idle
                [2, 0, 221, 223, 241, 323, 9],  # frame 9 - idle
                [2, 0, 223, 223, 243, 323, 10], # frame 10 - idle
            ],
            'video_info': {
                'width': 640,
                'height': 480,
                'fps': 30.0,
                'frame_count': 10,
                'path': 'test_video.mp4'
            }
        }
        
    def test_calculate_idle_time(self):
        """Test the calculation of idle time for employees"""
        # Call the analyze_tracking method with our sample data
        results = self.analytics.analyze_tracking(self.sample_tracks)
        
        # Check that idle_time is in the results
        self.assertIn('idle_time', results)
        
        # Check that we have idle time for both employees
        self.assertIn('1', results['idle_time'])
        self.assertIn('2', results['idle_time'])
        
        # Employee 1 should have less idle time than Employee 2
        self.assertLess(results['idle_time']['1'], results['idle_time']['2'])
        
        # Employee 2 should have significant idle time (at least 60% of their time)
        total_time = self.sample_tracks['video_info']['frame_count'] / self.sample_tracks['video_info']['fps']
        self.assertGreater(results['idle_time']['2'], 0.6 * total_time)
        
    def test_idle_period_detection(self):
        """Test the detection of idle periods"""
        # Call the analyze_tracking method with our sample data
        results = self.analytics.analyze_tracking(self.sample_tracks)
        
        # Check that idle_periods is in the results
        self.assertIn('idle_periods', results)
        
        # Check that we have idle periods for both employees
        self.assertIn('1', results['idle_periods'])
        self.assertIn('2', results['idle_periods'])
        
        # Employee 1 should have 1 idle period
        self.assertEqual(len(results['idle_periods']['1']), 1)
        
        # Employee 2 should have at least 2 idle periods
        self.assertGreaterEqual(len(results['idle_periods']['2']), 2)
        
if __name__ == '__main__':
    unittest.main() 