
import sys
import os
import unittest
from typing import List, Dict

# Add project root to path
sys.path.append(os.getcwd())

from src.api.main import consolidate_detections

class TestBBoxConsolidation(unittest.TestCase):
    def test_consolidation(self):
        # Two overlapping boxes
        # Box 1: [0.1, 0.1, 0.2, 0.2] (High confidence)
        # Box 2: [0.12, 0.12, 0.2, 0.2] (Lower confidence)
        detections = [
            {
                'class': 'Fire',
                'confidence': 0.9,
                'bbox': [0.1, 0.1, 0.2, 0.2],
                'source': 'yolo'
            },
            {
                'class': 'Fire',
                'confidence': 0.8,
                'bbox': [0.12, 0.12, 0.2, 0.2],
                'source': 'gpt'
            },
            {
                'class': 'Person', # Should not be consolidated by fire logic
                'confidence': 0.7,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'source': 'yolo'
            }
        ]
        
        consolidated = consolidate_detections(detections)
        
        # We expect 2 detections: the high-conf Fire and the Person
        self.assertEqual(len(consolidated), 2)
        
        # Check that the Fire detection kept is the highest confidence one
        fire_dets = [d for d in consolidated if d['class'] == 'Fire']
        self.assertEqual(len(fire_dets), 1)
        self.assertEqual(fire_dets[0]['confidence'], 0.9)
        self.assertEqual(fire_dets[0]['source'], 'yolo')
        
        # Check that Person is still there
        person_dets = [d for d in consolidated if d['class'] == 'Person']
        self.assertEqual(len(person_dets), 1)

    def test_no_overlap(self):
        # Two non-overlapping boxes
        detections = [
            {
                'class': 'Fire',
                'confidence': 0.9,
                'bbox': [0.1, 0.1, 0.1, 0.1],
                'source': 'yolo'
            },
            {
                'class': 'Fire',
                'confidence': 0.8,
                'bbox': [0.5, 0.5, 0.1, 0.1],
                'source': 'gpt'
            }
        ]
        
        consolidated = consolidate_detections(detections)
        self.assertEqual(len(consolidated), 2)

    def test_nested_boxes(self):
        # One small box inside a much larger box
        # Large: [0.1, 0.1, 0.5, 0.5] (Area 0.25)
        # Small: [0.2, 0.2, 0.1, 0.1] (Area 0.01)
        # Intersection: 0.01
        # Union: 0.25 + 0.01 - 0.01 = 0.25
        # IoU: 0.01 / 0.25 = 0.04 (Very low!)
        detections = [
            {
                'class': 'Fire',
                'confidence': 0.95,
                'bbox': [0.1, 0.1, 0.5, 0.5],
                'source': 'yolo'
            },
            {
                'class': 'Fire',
                'confidence': 0.9,
                'bbox': [0.2, 0.2, 0.1, 0.1],
                'source': 'gpt'
            }
        ]
        
        consolidated = consolidate_detections(detections)
        # With current IoU 0.5, this will return 2.
        # We WANT it to return 1.
        self.assertEqual(len(consolidated), 1)

if __name__ == '__main__':
    unittest.main()
