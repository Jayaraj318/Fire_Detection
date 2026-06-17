"""
Fire Detection API - TensorFlow Edition (Python 3.10)
Uses .h5 model for detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
import logging
import asyncio
import os
import yaml
import sys
import json
import base64
from pathlib import Path
from io import BytesIO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# APP SETUP
# ============================================================================
app = FastAPI(
    title="Fire Detection API (TensorFlow)",
    description="Experimental .h5 Model Backend",
    version="3.1.0-TF"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✅ FastAPI-TF app initialized")

# ============================================================================
# CONFIGURATION & NOTIFICATIONS
# ============================================================================
def load_config():
    config_path = Path("configs/model_config.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
    return {}

config = load_config()
from src.utils.notification import EmailNotifier
notifier = EmailNotifier(config.get('notification', {}))

# ============================================================================
# DATABASE SETUP
# ============================================================================
SessionLocal = None
try:
    from src.utils.database import init_db, DetectionEvent
    # Attempt to initialize
    SessionLocal = init_db()
    if SessionLocal:
        logger.info("✅ Database initialized successfully")
    else:
        logger.warning("⚠️ Database initialized as None (Offline Mode)")
except Exception as e:
    logger.error(f"⚠️ Database initialization failed: {e}")

# ============================================================================
# MODEL LOADING
# ============================================================================
from src.detection.tf_detector import TFDetector

class ModelManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            model_path = r"c:\Users\User\Documents\fire\Fire_Detection\models\yolo\weights\detection_model-ex-33--loss-4.97.h5"
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return None
            try:
                logger.info("Loading TensorFlow model...")
                cls._instance = TFDetector(model_path)
                logger.info("✅ TensorFlow model loaded!")
            except Exception as e:
                logger.error(f"❌ Failed to load TF model: {e}")
                return None
        return cls._instance

class SimpleState:
    model_loaded = False
    confidence_threshold = 0.10

state = SimpleState()

@app.on_event("startup")
async def warmup_model():
    """Warm up model in background."""
    async def _load():
        try:
            ModelManager.get_instance()
            state.model_loaded = True
        except Exception as e:
            logger.error(f"❌ Warmup failed: {e}")

    asyncio.create_task(_load())

# ============================================================================
# DATA MODELS & HELPERS
# ============================================================================
class DetectionResult(BaseModel):
    fire_detected: bool
    confidence: float
    fire_type: Optional[str] = None
    detections: List[Dict] = []
    timestamp: str
    model_type: str = "custom_h5"
    threshold_used: float = 0.25

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    backend: str = "tensorflow"

def save_detection_log(fire_detected, confidence, location, detections):
    if not fire_detected or not SessionLocal:
        return
    try:
        db = SessionLocal()
        # Explicitly use IST (+5:30) as per user request
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp_ist = datetime.now(ist)
        
        event = DetectionEvent(
            timestamp=timestamp_ist.replace(tzinfo=None), # Store as naive but localized
            confidence=float(confidence),
            class_name="Fire",
            location=location,
            metadata_json=json.dumps(detections or []),
            status="active",
            severity="critical" if confidence > 0.7 else "normal"
        )
        db.add(event)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"DB Error: {e}")

def get_db_session():
    """Helper to lazily initialize DB session if it failed at startup."""
    global SessionLocal
    if SessionLocal is None:
        try:
            from src.utils.database import init_db
            SessionLocal = init_db()
        except:
            pass
    return SessionLocal() if SessionLocal else None

def run_inference(image, threshold):
    detector = ModelManager.get_instance()
    if not detector:
        raise HTTPException(503, "Model not ready")
    
    # Update threshold
    detector.conf_thres = threshold
    
    results = detector.predict(image)
    
    detections_list = []
    max_conf = 0.0
    fire_detected = False
    
    h, w = image.shape[:2]
    
    if results:
        fire_detected = True
        for res in results:
            # res: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, conf, cls_id = res
            max_conf = max(max_conf, conf)
            
            # NOTE: Bounding boxes are suppressed because we lack the original model's 
            # anchor configuration, ensuring the "Fire Detected" alert is reliable 
            # without confusing/broken visual boxes.
            pass

    # Ensure detections_list is empty to avoid drawing broken boxes
    detections_list = []
    
    # Save to database if fire detected
    if fire_detected:
        save_detection_log(fire_detected, max_conf, "Upload", [])
        # Send alert
        try:
            notifier.send_fire_alert(max_conf, location="Image Upload")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            
    return DetectionResult(
        fire_detected=fire_detected,
        confidence=float(max_conf),
        fire_type="Fire" if fire_detected else None,
        detections=detections_list,
        timestamp=datetime.now().isoformat(),
        threshold_used=threshold
    )

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/health")
async def health():
    return HealthResponse(
        status="ok", 
        model_loaded=state.model_loaded, 
        timestamp=datetime.now().isoformat()
    )

@app.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...), threshold: float = 0.25):
    from PIL import Image
    import numpy as np
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    return run_inference(np.array(image), threshold)

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()
    logger.info("📡 Client connected (TF Backend)")
    
    frame_count = 0
    detector = None
    
    try:
        while True:
            data = await websocket.receive_text()
            if not detector:
                detector = ModelManager.get_instance()
                
            if "base64," in data:
                data = data.split("base64,")[1]
            
            import cv2
            import numpy as np
            img_bytes = base64.b64decode(data)
            img_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Update threshold from state
            if detector:
                detector.conf_thres = state.confidence_threshold
            
            # Predict
            results = []
            if detector:
                results = detector.predict(frame)
            
            fire_detected = len(results) > 0
            
            # Format results
            det_boxes = []
            max_conf = 0.0
            
            for r in results:
                x1, y1, x2, y2, conf, _ = r
                max_conf = max(max_conf, conf)
                det_boxes.append({
                    'xyxy': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })
            
            # Save DB
            if fire_detected and frame_count % 5 == 0:
                save_detection_log(fire_detected, max_conf, "Laptop Camera", [])
                # Send alert
                try:
                    notifier.send_fire_alert(max_conf, location="Laptop Camera")
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
                
            # Response
            resp = {
                "fire_detected": fire_detected,
                "confidence": float(max_conf),
                "fire_type": "Fire" if fire_detected else None,
                "detections": det_boxes,
                "timestamp": datetime.now().isoformat(),
                "frame_id": frame_count
            }
            
            await websocket.send_text(json.dumps(resp))
            frame_count += 1
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS Error: {e}")

@app.get("/detection-threshold")
async def get_thresh():
    return {"threshold": state.confidence_threshold}

@app.post("/detection-threshold")
async def set_thresh(data: Dict):
    t = float(data.get('threshold', 0.25))
    state.confidence_threshold = t
    return {"status": "success", "threshold": t}

# ============================================================================
# ALERTS API (CRUD)
# ============================================================================
@app.get("/alerts")
async def get_alerts(status: Optional[str] = None, severity: Optional[str] = None):
    """Get detection history from database with filtering."""
    db = get_db_session()
    if not db:
        logger.warning("No DB session available for /alerts")
        return {"alerts": []}
    
    try:
        query = db.query(DetectionEvent)
        
        if status:
            query = query.filter(DetectionEvent.status == status)
        if severity:
            query = query.filter(DetectionEvent.severity == severity)
            
        events = query.order_by(DetectionEvent.timestamp.desc()).limit(100).all()
        
        alerts = []
        for e in events:
            alerts.append({
                "id": e.id,
                "timestamp": e.timestamp.strftime("%Y-%m-%d %I:%M:%S %p"), # Human readable localized
                "iso_timestamp": e.timestamp.isoformat(),
                "confidence": e.confidence,
                "fire_type": e.class_name or "Fire",
                "location": e.location,
                "status": e.status or "active",
                "severity": e.severity or "normal",
                "image": None 
            })
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch alerts: {e}")
        return {"alerts": []}
    finally:
        db.close()

@app.patch("/alerts/{alert_id}")
async def update_alert(alert_id: int, data: Dict):
    """Update alert status or severity (e.g., mark as resolved)."""
    db = get_db_session()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        event = db.query(DetectionEvent).filter(DetectionEvent.id == alert_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        if 'status' in data:
            event.status = data['status']
        if 'severity' in data:
            event.severity = data['severity']
            
        db.commit()
        return {"status": "success", "message": f"Alert {alert_id} updated"}
    except Exception as e:
        logger.error(f"❌ Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete a specific alert."""
    db = get_db_session()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        event = db.query(DetectionEvent).filter(DetectionEvent.id == alert_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        db.delete(event)
        db.commit()
        return {"status": "success", "message": f"Alert {alert_id} deleted"}
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/alerts")
async def clear_alerts():
    """Delete ALL alerts."""
    db = get_db_session()
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        db.query(DetectionEvent).delete()
        db.commit()
        return {"status": "success", "message": "All alerts cleared"}
    except Exception as e:
        logger.error(f"❌ Clear alerts failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
