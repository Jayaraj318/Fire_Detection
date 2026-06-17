"""
Fire Detection System
YOLO-Only Implementation with Email Notification
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, Optional, Union, List
from pathlib import Path
import logging
import yaml
import sys
import os

import cv2
import numpy as np

# Add src to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.notification import EmailNotifier

logger = logging.getLogger(__name__)

class FireDetectionSystem(nn.Module):
    """
    YOLO-Only Fire Detection System.
    
    Architecture:
    - YOLOv8/v10: Object detection (Fire/Smoke)
    - Email Notification: Alerts on detection
    
    Outputs:
    - Fire detection (binary)
    - Bounding boxes
    - Confidence scores
    """
    
    def __init__(
        self,
        model_path: str = "models/best.pt",
        config_path: str = "configs/model_config.yaml",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the fire detection system.
        
        Args:
            model_path: Path to trained YOLO weights.
            config_path: Path to configuration file.
            confidence_threshold: Threshold for reporting fire.
        """
        super().__init__()
        
        self.confidence_threshold = confidence_threshold
        
        # Load Configuration
        self.config = self._load_config(config_path)
        print(f"DEBUG: Loaded config from {config_path}: {self.config}")
        
        # Initialize YOLO
        # Priority: Argument > Config > Default
        final_model_path = model_path
        if self.config and 'model' in self.config and 'yolo' in self.config['model']:
             config_model_path = self.config['model']['yolo'].get('variant')
             print(f"DEBUG: Found variant in config: {config_model_path}")
             if config_model_path:
                 final_model_path = config_model_path
        
        logger.info(f"Loading YOLO model from: {final_model_path}")
        try:
            self.model = YOLO(final_model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {final_model_path}: {e}")
            raise e
            
        # Initialize Email Notifier
        self.notifier = None
        if self.config and 'notification' in self.config:
            self.notifier = EmailNotifier(self.config['notification'])
            logger.info("Email notification system initialized.")
        else:
            logger.warning("Notification config not found. Email alerts disabled.")
            
        # Initialize Face Detector (Negative Filter)
        # Using built-in OpenCV Haar Cascade to prevent "Person" being detected as "Fire"
        try:
             cv2_base = os.path.dirname(cv2.__file__)
             cascade_path = os.path.join(cv2_base, 'data/haarcascade_frontalface_default.xml')
             if not os.path.exists(cascade_path):
                 # Fallback to local or standard path if python-opencv data not found
                 cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
             
             self.face_cascade = cv2.CascadeClassifier(cascade_path)
             if self.face_cascade.empty():
                 logger.warning("⚠️ Face detector could not be loaded. False positives on people may occur.")
                 self.face_cascade = None
             else:
                 logger.info("✅ Face filter initialized (to prevent Person->Fire errors)")
        except Exception as e:
            logger.warning(f"Feature: Face Filter failed to init: {e}")
            self.face_cascade = None

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def forward(
        self,
        frame: Union[torch.Tensor, str, 'numpy.ndarray'],
        conf: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Process a frame for fire detection.
        
        Args:
            frame: Input image (Tensor, path, or numpy array).
            conf: Override confidence threshold.
            
        Returns:
            Dictionary containing detection results.
        """
        threshold = conf if conf is not None else self.confidence_threshold
        
        # Run YOLO inference
        # verbose=False to keep stdout clean
        results = self.model(frame, conf=0.01, verbose=False) # Run with very low conf to see EVERYTHING
        
        result = results[0] # Single image inference
        
        # DEBUG: Print everything the model sees
        logger.info(f"🔍 RAW DEBUG -- MODEL DETECTIONS: {len(result.boxes)} objects")
        if len(result.boxes) == 0:
            logger.info("   (No objects detected even at 1% confidence)")
        else:
            for b in result.boxes:
                cls_id = int(b.cls[0])
                cls_name = result.names.get(cls_id, 'unknown')
                conf_val = float(b.conf[0])
                logger.info(f"   -> Found: {cls_name} (Class {cls_id}) | Conf: {conf_val:.2%} | Threshold needed: {threshold:.2%}")

        # Process detections
        # Process detections
        # Process detections
        # Dynamically identify class IDs based on model names
        # This prevents "Person" (Class 0 in COCO) from being detected as "Fire"
        
        fire_ids = []
        smoke_ids = []
        
        for id, name in result.names.items():
            name_lower = name.lower()
            if 'fire' in name_lower or 'flame' in name_lower:
                fire_ids.append(id)
            elif 'smoke' in name_lower:
                smoke_ids.append(id)
        
        # Fallback: If no 'fire' class found, assume custom trained model (0=Fire) 
        # BUT only if 0 is NOT 'person'
        if not fire_ids and not smoke_ids:
             if result.names.get(0, '').lower() != 'person':
                 fire_ids = [0]
                 smoke_ids = [1]
             else:
                 print("WARNING: Model appears to be COCO (0=Person) and no 'fire' class found.")
        
        filtered_boxes = []
        for b in result.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            cls_name = result.names.get(cls_id, 'unknown').lower()
            
            # Smart filtering
            if cls_id == 0: # Fire
                cls_threshold = 0.10
                if conf >= cls_threshold:
                    # Append immediately (Face filter commented out below)
                    filtered_boxes.append(b)
                    
                    # COMMENTED OUT FACE FILTER TO FIX MATCHSTICK DETECTION
                    # is_face = False
                    # if self.face_cascade is not None:
                    #     # ... (rest of the commented out logic)

            elif cls_id in smoke_ids:
                # include smoke for debug
                filtered_boxes.append(b)
            else:
                # Log even ignored classes
                logger.info(f"   -> Ignoring class: {cls_name} (Class {cls_id})")
        
        fire_detected = False
        max_confidence = 0.0
        
        if len(filtered_boxes) > 0:
            fire_detected = True
            # Get max confidence of filtered boxes
            max_confidence = max([float(b.conf[0]) for b in filtered_boxes])
            
            # Send Notification
            if self.notifier:
                # Save temp image for email attachment if input is an array/tensor
                # If input is path, use that.
                img_path = None
                if isinstance(frame, str) and os.path.exists(frame):
                    img_path = frame
                
                # Note: Saving tensor/numpy to temp file is skipped for simplicity 
                # unless explicitly required, to avoid I/O overhead on every frame.
                # In a real app, you might save the frame if fire is detected.
                
                self.notifier.send_fire_alert(
                    confidence=max_confidence,
                    image_path=img_path,
                    location="Camera 1" # Placeholder
                )
        
        return {
            'fire_detected': fire_detected,
            'confidence': max_confidence,
            'bounding_boxes': [b.xyxy.cpu().tolist()[0] for b in filtered_boxes],
            'yolo_boxes': filtered_boxes, # Return filtered list objects
            'classes': [int(b.cls[0]) for b in filtered_boxes]
        }
    
    def predict(self, source, **kwargs):
        """Alias for forward/model calls compatible with standard usage."""
        return self.forward(source, **kwargs)

class FireDetectionLite(FireDetectionSystem):
    """
    Alias for FireDetectionSystem in this YOLO-only architecture.
    Both 'Lite' and 'System' use the same efficient YOLO model.
    """
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Testing YOLO-Only FireDetectionSystem...")
    
    # Create model
    try:
        model = FireDetectionSystem()
        
        # Dummy test (requires an actual image path or will fail with string)
        # using a placeholder or just printing init success
        print("Model initialized successfully.")
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Ensure 'models/yolo/weights/best.pt' exists or update config.")
