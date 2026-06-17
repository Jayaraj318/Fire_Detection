import os
import sys
import numpy as np
from PIL import Image
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.detection.gemini_detector import GeminiFireDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detector():
    api_key = "AIzaSyBt3iLher5_bLMVEax2febw5vHNQLmXVrw"
    detector = GeminiFireDetector(api_key=api_key)
    
    # Create a dummy "bright spot" image (simulating a match in dark)
    # 640x480 black image with a bright orange/white circle
    img_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2
    cv2.circle(img_arr, (320, 240), 20, (255, 255, 255), -1) # White core
    cv2.circle(img_arr, (320, 240), 40, (0, 165, 255), 5)    # Orange glow
    
    logger.info("Testing Gemini 2.0 detection...")
    result = detector.detect(img_arr)
    
    print("\n" + "="*50)
    print("DETECTION RESULT:")
    print(f"Fire Detected: {result.get('fire_detected')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Type: {result.get('fire_type')}")
    print(f"Description: {result.get('description')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_detector()
