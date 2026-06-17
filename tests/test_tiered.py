import os
import sys
import numpy as np
from PIL import Image
import logging
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.api.main import process_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tiered_detection():
    # Create a dummy image
    img_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a bright spot to simulate a match
    import cv2
    cv2.circle(img_arr, (320, 240), 20, (255, 255, 255), -1)
    cv2.circle(img_arr, (320, 240), 40, (0, 165, 255), 5)
    
    logger.info("🧪 Test 1: Full AI Tiered Detection (Throttled frame equivalent)")
    res1 = process_inference(img_arr, threshold=0.25, use_ai=True)
    print(f"Result 1: Detected={res1.fire_detected}, Confidence={res1.confidence}, Type={res1.fire_type}, Model={res1.model_type}")
    
    logger.info("🧪 Test 2: YOLO Only (Regular frame equivalent)")
    res2 = process_inference(img_arr, threshold=0.25, use_ai=False)
    print(f"Result 2: Detected={res2.fire_detected}, Confidence={res2.confidence}, Type={res2.fire_type}, Model={res2.model_type}")

if __name__ == "__main__":
    test_tiered_detection()
