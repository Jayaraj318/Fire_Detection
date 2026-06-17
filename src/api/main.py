"""
Fire Detection API - Railway-Compatible
FastAPI backend using YOLO-Only model for fire detection
- Fast startup: NO ML models loaded at import time
- Lazy loading: Models load only on first API/WebSocket call
- CPU-only: Defaults to CPU, no CUDA requirement
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
import json
from io import BytesIO
import base64
import os
import asyncio
import cv2
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging to use uvicorn logger
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# ============================================================================
# FAST STARTUP: Initialize FastAPI app immediately (NO heavy ML imports yet)
# ============================================================================
app = FastAPI(
    title="Fire Detection API",
    description="YOLO-Only Fire Detection System (Railway-Compatible)",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✅ FastAPI app initialized (ready to bind to $PORT)")

# ============================================================================
# DATABASE INITIALIZATION (lightweight, happens at import time)
# ============================================================================
try:
    from src.utils.database import init_db, DetectionEvent, SystemConfig, SessionLocal
    logger.info("✅ Database module imported")
    SessionLocal = init_db()
    logger.info("✅ Database initialized")
except Exception as e:
    logger.error(f"⚠️  Database initialization failed: {e}")
    SessionLocal = None

# ============================================================================
# ============================================================================
# MODEL LOADING - Now handled by src/detection/model_loader.py
# Using singleton pattern - model loads ONCE on first use
# ============================================================================
from src.detection.model_loader import get_model, get_detector_system, get_gemini_detector, get_gpt_detector

GEMINI_API_KEY = "AIzaSyBt3iLher5_bLMVEax2febw5vHNQLmXVrw"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# RATE LIMITING STATE
_last_ai_time = 0
AI_COOLDOWN = 1.8 # Slightly under 2s for better responsiveness

def get_address_from_coords(lat: float, lng: float) -> Optional[str]:
    """Resolve latitude and longitude to a human-readable address."""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="fire_detection_app")
        location = geolocator.reverse((lat, lng), language='en')
        return location.address if location else None
    except Exception as e:
        logger.error(f"❌ Geocoding failed: {e}")
        return None

# Simple state tracker for health checks
class SimpleState:
    model_loaded = False
    confidence_threshold = 0.10  # Lowered to 0.10 to catch small/distant fires

state = SimpleState()


import asyncio


@app.on_event("startup")
async def warmup_model():
    """Warm up YOLO model in background after startup to avoid blocking first request on Render."""
    async def _load():
        try:
            logger.info("🔥 Warming up YOLO model in background...")
            # Load model in a thread to avoid blocking event loop
            await asyncio.to_thread(get_model)
            state.model_loaded = True
            logger.info("✅ YOLO model warmed up (background)")
        except Exception as e:
            logger.error(f"❌ Model warm-up failed: {e}")

    # Schedule background warm-up task; do not await it here
    asyncio.create_task(_load())


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class DetectionResult(BaseModel):
    fire_detected: bool
    confidence: float
    fire_type: Optional[str] = None
    detections: List[Dict] = []
    bounding_boxes: List[List[float]] = [] # For frontend (x, y, w, h normalized)
    timestamp: str
    model_type: str = "yolo"
    threshold_used: float = 0.5
    yolo_suspect: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# ============================================================================
# LIGHTWEIGHT HEALTH CHECK (fast startup, no ML)
# ============================================================================
@app.get("/health")
async def health_check() -> HealthResponse:
    """
    Lightweight health check - returns immediately.
    Does NOT load ML models.
    """
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# DETECTION HELPER FUNCTIONS
# ============================================================================
def consolidate_detections(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Consolidate overlapping detections using Non-Maximum Suppression (NMS) logic.
    Merges YOLO and AI detections that refer to the same fire source.
    """
    if not detections:
        return []

    # Filter for fire-related detections
    fire_dets = [d for d in detections if d.get('class', '').lower() in [
        'fire', 'matchstick', 'candle', 'lighter', 'large fire', 'smoke', 'flame', 'hazard', 'fire_hazard'
    ]]
    other_dets = [d for d in detections if d not in fire_dets]
    
    if not fire_dets:
        return detections

    # Sort by confidence descending
    fire_dets.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while fire_dets:
        best = fire_dets.pop(0)
        keep.append(best)
        
        # Compare with remaining detections
        remaining = []
        for det in fire_dets:
            # Calculate IoU
            # bbox is [x, y, w, h] normalized
            b1 = best['bbox']
            b2 = det['bbox']
            
            # Intersection
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[0] + b1[2], b2[0] + b2[2])
            inter_y2 = min(b1[1] + b1[3], b2[1] + b2[3])
            
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            # Union
            area1 = b1[2] * b1[3]
            area2 = b2[2] * b2[3]
            union_area = area1 + area2 - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            # Intersection over Minimum (IoM) - Good for nested boxes
            min_area = min(area1, area2)
            iom = inter_area / min_area if min_area > 0 else 0
            
            # Merge if high overlap OR one box is largely inside the other
            if iou < iou_threshold and iom < 0.8:
                remaining.append(det)
            else:
                logger.info(f"Merging overlapping {det['source']} box into {best['source']} (IoU: {iou:.2f}, IoM: {iom:.2f})")
                
        fire_dets = remaining
        
    return keep + other_dets


def save_detection_to_db(
    fire_detected: bool,
    confidence: float,
    location: str = "Unknown",
    detections_list: List[Dict] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None
):
    """Save detection event to database with error handling."""
    if not fire_detected or not SessionLocal:
        return

    try:
        db = SessionLocal()
        event = DetectionEvent(
            timestamp=datetime.utcnow(),
            confidence=float(confidence),
            class_name="Fire",
            location=location,
            lat=lat,
            lng=lng,
            metadata_json=json.dumps(detections_list or [])
        )
        db.add(event)
        db.commit()
        db.close()
        logger.debug(f"💾 Saved detection to DB: confidence={confidence:.2%}, location={location}")
    except Exception as e:
        logger.error(f"❌ Failed to save to DB: {e}")


def process_inference(
    image_np, 
    threshold: float = 0.5, 
    lat: Optional[float] = None, 
    lng: Optional[float] = None, 
    force_ai: bool = False, 
    location_name: Optional[str] = None,
    skip_ai: bool = False
) -> DetectionResult:
    """
    Run tiered fire detection.
    """
    logger.debug(f"⚙️ process_inference called (force_ai={force_ai}, skip_ai={skip_ai})")
    global _last_ai_time
    # 0. Defensive Location Cleaning & Debugging
    def clean_telemetry(val):
        if val is None: return None
        s_val = str(val).strip()
        if s_val.lower() in ('none', 'null', 'undefined', ''): return None
        return s_val

    location_name = clean_telemetry(location_name)
    
    # Ensure lat/lng are actual numbers or None
    try:
        if lat is not None and str(lat).lower() != 'none': lat = float(lat)
        else: lat = None
    except (ValueError, TypeError): lat = None
    
    try:
        if lng is not None and str(lng).lower() != 'none': lng = float(lng)
        else: lng = None
    except (ValueError, TypeError): lng = None

    logger.info(f"📍 Location Telemetry: lat={lat} ({type(lat).__name__}), "
                f"lng={lng} ({type(lng).__name__}), "
                f"location_name='{location_name}' ({type(location_name).__name__})")
    
    # 1. Resolve Address & Image Specs
    address = None
    if lat is not None and lng is not None:
        address = get_address_from_coords(lat, lng)
    
    h, w = image_np.shape[:2]

    # 2. YOLO Detection (Fast Baseline)
    model = get_model()
    yolo_detected = False
    yolo_conf = 0.0
    detections_list = []
    yolo_suspect = False # Flag for low-confidence YOLO find to trigger AI

    if model:
        image_bgr = image_np[:, :, ::-1]
        # Use a very low conf for YOLO internally to see ALL candidates
        yolo_results = model.predict(image_bgr, conf=0.05, verbose=False)
        
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            n_boxes = len(result.boxes) if result.boxes is not None else 0
            logger.info(f"🔬 YOLO: {n_boxes} raw boxes found at conf>=0.05")
            if n_boxes > 0:
                for b in result.boxes:
                    try:
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0])
                        logger.info(f"  📦 cls={cls_id} conf={conf:.3f}")
                        
                        # ONLY PROCESS FIRE (Class 0)
                        if cls_id == 0:
                            yolo_suspect = True
                            if conf >= threshold:
                                yolo_detected = True
                                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                                yolo_conf = max(yolo_conf, conf)
                                detections_list.append({
                                    'class': 'fire',
                                    'confidence': conf,
                                    'bbox': [float(xyxy[0])/w, float(xyxy[1])/h, float(xyxy[2]-xyxy[0])/w, float(xyxy[3]-xyxy[1])/h],
                                    'xyxy': [int(x) for x in xyxy],
                                    'source': 'yolo'
                                })
                            else:
                                logger.info(f"  ⚠️ Fire candidate ignored (conf {conf:.3f} < threshold {threshold:.3f})")
                    except Exception as e:
                        logger.error(f"Box parse error: {e}")
                        continue
            else:
                logger.info("  ⚪ YOLO found nothing (0 boxes)")

    # 3. Fast YOLO path — return detection to UI immediately
    if skip_ai:
        if yolo_detected:
            clean_loc = location_name if location_name and location_name.lower() not in ('none', '') else None
            if address:
                display_location = f"{address} [{clean_loc}]" if clean_loc else address
            elif clean_loc:
                display_location = clean_loc
            else:
                display_location = "Field Detection Unit"

            logger.info(f"📢 YOLO Alert: '{display_location}' ({yolo_conf:.1%})")
            save_detection_to_db(yolo_detected, yolo_conf, display_location, detections_list, lat, lng)

            global _last_ai_time
            current_time = time.time()
            can_send_email = (current_time - _last_ai_time) >= AI_COOLDOWN

            if can_send_email:
                # YOLO detection triggers email directly (cooldown-guarded)
                _last_ai_time = current_time
                detector = get_detector_system()
                if detector and detector.notifier:
                    temp_path = None
                    try:
                        temp_dir = Path("temp_alerts")
                        temp_dir.mkdir(exist_ok=True)
                        import tempfile
                        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=str(temp_dir))
                        temp_path = temp_file.name
                        temp_file.close()
                        cv2.imwrite(temp_path, image_bgr)
                        detector.notifier.send_fire_alert(
                            confidence=yolo_conf,
                            image_path=temp_path,
                            location=display_location,
                            lat=lat, lng=lng, address=address
                        )
                        logger.info("✅ Email alert sent (YOLO only)")
                    except Exception as notify_err:
                        logger.error(f"❌ Email failed: {notify_err}")
                    finally:
                        if temp_path:
                            try: Path(temp_path).unlink(missing_ok=True)
                            except: pass
            else:
                logger.info(f"⏳ Email cooldown active ({AI_COOLDOWN}s), skipping email")

        return DetectionResult(
            fire_detected=yolo_detected,
            confidence=float(yolo_conf),
            fire_type="Fire" if yolo_detected else None,
            detections=detections_list,
            bounding_boxes=[d['bbox'] for d in detections_list],
            timestamp=datetime.now().isoformat(),
            model_type="yolo",
            threshold_used=threshold,
            yolo_suspect=yolo_suspect
        )

    # 4. Tiered AI Detection (Triggered by force_ai OR yolo_suspect)
    ai_result = {"fire_detected": False, "confidence": 0.0, "fire_type": None, "source": "none"}
    
    current_time = time.time()
    can_call_ai = (current_time - _last_ai_time) >= AI_COOLDOWN
    
    if (force_ai or yolo_suspect) and can_call_ai:
        _last_ai_time = current_time
        logger.info(f"🔍 System triggered AI Check (Force: {force_ai}, YOLO Suspect: {yolo_suspect})")
        
        # TIER 1: GEMINI
        gemini_detector = get_gemini_detector(api_key=GEMINI_API_KEY)
        if gemini_detector:
            try:
                gemini_res = gemini_detector.detect(image_np)
                if gemini_res.get("fire_detected"):
                    # ONLY PROCESS FIRE (Exclude Smoke if strictly restricted to fire)
                    if gemini_res.get("fire_type") != "Smoke":
                        ai_result = {**gemini_res, "source": "gemini"}
                    else:
                        logger.info("💨 Gemini detected Smoke, but filtering for FIRE ONLY.")
            except Exception as e:
                logger.warning(f"Gemini Tier failed: {e}")

        if not ai_result["fire_detected"]:
            gpt_detector = get_gpt_detector(api_key=OPENAI_API_KEY)
            if gpt_detector:
                try:
                    gpt_res = gpt_detector.detect(image_np)
                    if gpt_res.get("fire_detected"):
                        # ONLY PROCESS FIRE (Exclude Smoke if strictly restricted to fire)
                        if gpt_res.get("fire_type") != "Smoke":
                            ai_result = {**gpt_res, "source": "gpt"}
                        else:
                            logger.info("💨 GPT detected Smoke, but filtering for FIRE ONLY.")
                except Exception as e:
                    logger.warning(f"GPT Tier failed: {e}")

    # 4. Hybrid Bounding Boxes: If AI found it but YOLO didn't, add AI boxes
    if ai_result["fire_detected"] and ai_result.get("bboxes"):
        for bbox in ai_result["bboxes"]:
            try:
                # Gemini/GPT use [ymin, xmin, ymax, xmax] 0-1
                ymin, xmin, ymax, xmax = [float(x) for x in bbox]
                detections_list.append({
                    'class': ai_result.get("fire_type", "Fire"),
                    'confidence': ai_result.get("confidence", 0.9),
                    'bbox': [xmin, ymin, xmax-xmin, ymax-ymin], # [x, y, w, h] 0-1
                    'xyxy': [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)],
                    'source': ai_result["source"]
                })
            except: continue

    # 5. Consolidate Bounding Boxes (NMS)
    if detections_list:
        detections_list = consolidate_detections(detections_list)

    # 6. Fusion Logic
    fire_detected = ai_result["fire_detected"] or yolo_detected
    confidence = max(ai_result["confidence"], yolo_conf)
    fire_type = ai_result["fire_type"] if ai_result["fire_detected"] else ("Fire" if yolo_detected else None)
    model_type = f"tiered({ai_result['source']}+yolo)" if ai_result["source"] != "none" else "yolo"

    # 7. Trigger Alerts
    if fire_detected:
        # Clean 'None' strings from frontend
        clean_loc = location_name if location_name and location_name.lower() not in ('none', '') else None

        # GPS address takes priority; custom device label is secondary identifier
        if address:
            display_location = f"{address} [{clean_loc}]" if clean_loc else address
        elif clean_loc:
            display_location = clean_loc
        else:
            display_location = "Field Detection Unit"
            
        logger.info(f"📢 Final reporting location: '{display_location}'")
        save_detection_to_db(fire_detected, confidence, display_location, detections_list, lat, lng)
        
        detector = get_detector_system()
        if detector and detector.notifier:
            # Send alert when AI confirmed OR YOLO detected at threshold
            if ai_result["fire_detected"] or yolo_detected:
                # Save temp image for email
                temp_path = None
                try:
                    import tempfile
                    temp_dir = Path("temp_alerts")
                    temp_dir.mkdir(exist_ok=True)
                    
                    # Convert RGB (image_np is RGB) to BGR for cv2
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=str(temp_dir))
                    temp_path = temp_file.name
                    temp_file.close() # Close so cv2 can write
                    
                    cv2.imwrite(temp_path, image_bgr)
                    
                    detector.notifier.send_fire_alert(
                        confidence=confidence,
                        image_path=temp_path,
                        location=display_location,
                        lat=lat, lng=lng, address=address
                    )
                except Exception as notify_err:
                    logger.error(f"❌ Notification failed: {notify_err}")
                finally:
                    if temp_path:
                        try:
                            file_to_del = Path(temp_path)
                            if file_to_del.exists():
                                file_to_del.unlink()
                        except: pass

    return DetectionResult(
        fire_detected=fire_detected,
        confidence=float(confidence),
        fire_type=fire_type,
        detections=detections_list,
        bounding_boxes=[d['bbox'] for d in detections_list],
        timestamp=datetime.now().isoformat(),
        model_type=model_type,
        threshold_used=threshold,
        yolo_suspect=yolo_suspect
    )


# ============================================================================
# REST API ENDPOINTS
# ============================================================================
@app.post("/detect", response_model=DetectionResult)
async def detect_fire(
    file: UploadFile = File(...),
    threshold: float = Form(0.25),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    location_name: Optional[str] = Form(None)
):
    """
    Upload image for fire detection.
    """
    try:
        from PIL import Image
        import numpy as np
        
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        return process_inference(image_np, threshold, lat, lng, location_name=location_name)
        
    except Exception as e:
        logger.error(f"❌ Upload detection failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect/base64", response_model=DetectionResult)
async def detect_fire_base64(data: Dict):
    """Upload base64-encoded image for fire detection."""
    try:
        from PIL import Image
        import numpy as np
        
        image_data = data.get('image', '')
        threshold = float(data.get('threshold', 0.5))
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        lat = data.get('lat')
        lng = data.get('lng')
        location_name = data.get('location_name')
        
        return process_inference(image_np, threshold, lat, lng, location_name=location_name)
        
    except Exception as e:
        logger.error(f"❌ Base64 detection failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")


@app.post("/test/thresholds")
async def test_thresholds(file: UploadFile = File(...)):
    """Test detection with multiple confidence thresholds."""
    try:
        from PIL import Image
        import numpy as np
        
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for thresh in thresholds:
            res = process_inference(image_np, threshold=thresh)
            results[f"threshold_{thresh}"] = {
                "fire_detected": res.fire_detected,
                "confidence": res.confidence,
                "status": "🔥 FIRE" if res.fire_detected else "✅ SAFE"
            }
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"❌ Threshold test failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/alerts")
async def get_alerts():
    """Get detection history from database."""
    if not SessionLocal:
        return {"alerts": []}
    
    db = SessionLocal()
    try:
        events = db.query(DetectionEvent).order_by(
            DetectionEvent.timestamp.desc()
        ).limit(50).all()
        
        alerts = []
        for e in events:
            alerts.append({
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "confidence": e.confidence,
                "location": e.location,
                "lat": e.lat,
                "lng": e.lng,
                "status": e.status or "active",
                "severity": e.severity or "normal"
            })
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch alerts: {e}")
        return {"alerts": []}
    finally:
        db.close()
        

@app.patch("/alerts/{alert_id}")
async def update_alert_status(alert_id: int, status_update: dict):
    """Update alert status (e.g. resolve)."""
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="Database not available")
    
    status = status_update.get("status")
    if not status:
        raise HTTPException(status_code=400, detail="Missing status in request body")
        
    db = SessionLocal()
    try:
        event = db.query(DetectionEvent).filter(DetectionEvent.id == alert_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        event.status = status
        db.commit()
        logger.info(f"✅ Updated alert {alert_id} status to {status}")
        return {"status": "success", "message": f"Alert {alert_id} updated to {status}"}
    except Exception as e:
        logger.error(f"❌ Failed to update alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete a detection alert by ID."""
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="Database not available")
    
    db = SessionLocal()
    try:
        event = db.query(DetectionEvent).filter(DetectionEvent.id == alert_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        db.delete(event)
        db.commit()
        logger.info(f"🗑️ Deleted alert {alert_id}")
        return {"status": "success", "message": f"Alert {alert_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/detection-threshold")
async def get_detection_threshold():
    """Get current fire detection confidence threshold."""
    return {
        "threshold": state.confidence_threshold,
        "description": "Minimum confidence (0.0-1.0) for fire detection"
    }


@app.post("/detection-threshold")
async def set_detection_threshold(data: Dict):
    """Set fire detection confidence threshold."""
    try:
        threshold = float(data.get('threshold', 0.5))
        
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be 0.0-1.0")
        
        state.confidence_threshold = threshold
        logger.info(f"🔧 Detection threshold updated to {threshold}")
        
        return {
            "status": "success",
            "threshold": state.confidence_threshold
        }
        
    except Exception as e:
        logger.error(f"❌ Threshold update failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/notification-status")
async def get_notification_status():
    """Get notification system status."""
    detector = get_detector_system()
    
    if not detector or not hasattr(detector, 'notifier') or not detector.notifier:
        return {
            "status": "not_configured",
            "message": "Notification system not available"
        }
    
    try:
        status = detector.notifier.get_status()
        return {
            "status": "ok",
            "notifications": status
        }
    except Exception as e:
        logger.error(f"❌ Notification status failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/test-notification")
async def test_notification():
    """Send test notification."""
    detector = get_detector_system()
    
    if not detector or not hasattr(detector, 'notifier') or not detector.notifier:
        raise HTTPException(status_code=503, detail="Notification system not available")
    
    try:
        detector.notifier.send_fire_alert(
            confidence=0.95,
            location="TEST - Fire Detection System",
            image_path=None
        )
        
        status = detector.notifier.get_status()
        return {
            "status": "success",
            "message": "Test notification sent",
            "details": {
                "email_configured": status.get("email_configured", False),
                "sms_configured": status.get("sms_configured", False)
            }
        }
    except Exception as e:
        logger.error(f"❌ Test notification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NON-BLOCKING AI HELPERS
# ============================================================================
async def background_ai_verification(websocket: WebSocket, frame_rgb, lat, lng, location_name):
    """
    Runs the deep AI check in the background. Does not block the main loop.
    Gracefully handles the case where the WebSocket is closed before it finishes.
    """
    try:
        res = await asyncio.to_thread(
            process_inference,
            frame_rgb,
            threshold=state.confidence_threshold,
            lat=lat, lng=lng,
            force_ai=True,
            location_name=location_name,
            skip_ai=False
        )
        
        if res.fire_detected:
            logger.info(f"✅ Background AI verified fire: {res.fire_type} ({res.confidence:.1%})")
        else:
            logger.info("ℹ️ Background AI cleared the suspect area.")

        # Only send if WebSocket is still alive
        try:
            update_data = {
                "type": "verification",
                "fire_detected": res.fire_detected,
                "confidence": float(res.confidence),
                "fire_type": res.fire_type,
                "detections": res.detections,
                "timestamp": res.timestamp,
                "model_type": res.model_type
            }
            await websocket.send_text(json.dumps(update_data))
        except Exception:
            logger.debug("🔌 WebSocket closed before background AI could report. Result discarded.")
            
    except Exception as e:
        logger.error(f"❌ Background AI task failed: {e}")


# ============================================================================
# WEBSOCKET ENDPOINTS - LAZY LOADING SAFE
# ============================================================================
@app.websocket("/ws/video")
async def client_stream_detection(websocket: WebSocket):
    """
    Client-side WebSocket (Laptop Camera Mode).
    Receives base64 frames → Runs inference → Returns detections.
    
    Model is lazy-loaded on first frame.
    """
    await websocket.accept()
    logger.info("✅ WebSocket accepted. Waiting for first frame...")
    is_connected = True
    detector = None
    
    logger.info("📡 Client WebSocket connected")
    frame_count = 0
    
    try:
        while is_connected:
            try:
                # 1. Receive data
                data = await websocket.receive_text()
                if not data:
                    continue
                
                # 2. Lazy load detector
                if detector is None:
                    logger.info("⏳ Loading detector system for live stream...")
                    detector = get_detector_system()
                    if not detector:
                        await websocket.send_text(json.dumps({"error": "Detector system failed to load"}))
                        break
                    state.model_loaded = True

                # 3. Process Frame
                try:
                    # Parse JSON or raw base64
                    lat, lng = None, None
                    location_name = None
                    if "{" in data[:20]: # Heuristic for JSON
                        try:
                            json_data = json.loads(data)
                            data = json_data.get('image', data)
                            lat = json_data.get('lat')
                            lng = json_data.get('lng')
                            location_name = json_data.get('location_name')
                        except: pass
                    
                    if "base64," in data:
                        data = data.split("base64,")[1]
                    
                    img_bytes = base64.b64decode(data)
                    img_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        logger.error("⚠️ Failed to decode frame")
                        continue

                    if frame_count == 0:
                        logger.info("🎬 First frame successfully received and decoded!")

                    # Heartbeat log
                    if frame_count % 30 == 0:
                        logger.info(f"📥 Received frame {frame_count}")

                    # 4. FAST YOLO INFERENCE (Non-blocking)
                    # Run with skip_ai=True for instant results
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res_obj = await asyncio.to_thread(
                        process_inference,
                        rgb_frame, 
                        threshold=state.confidence_threshold, 
                        lat=lat, lng=lng, 
                        force_ai=False,
                        location_name=location_name,
                        skip_ai=True
                    )

                    # 5. INSTANT REPORT - YOLO only, no AI background tasks
                    fire_detected = res_obj.fire_detected
                    if fire_detected:
                        logger.warning(f"🔥 FIRE DETECTED: {res_obj.fire_type} ({res_obj.confidence:.1%})")

                    response = {
                        "type": "detection",
                        "fire_detected": fire_detected,
                        "confidence": float(res_obj.confidence),
                        "fire_type": res_obj.fire_type or ("Fire" if fire_detected else None),
                        "detections": res_obj.detections,
                        "yolo_suspect": res_obj.yolo_suspect,
                        "model_type": "yolo",
                        "timestamp": datetime.now().isoformat(),
                        "frame_id": frame_count
                    }
                    await websocket.send_text(json.dumps(response))
                    frame_count += 1

                except WebSocketDisconnect:
                    raise  # Bubble up to outer handler
                except Exception as inner_e:
                    logger.error(f"❌ WebSocket loop error on frame {frame_count}: {inner_e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    frame_count += 1
                    continue

            except WebSocketDisconnect:
                logger.info("📡 Client disconnected")
                is_connected = False
                break
            except Exception as outer_e:
                err_msg = str(outer_e).lower()
                logger.error(f"❌ Critical WebSocket error: {outer_e}")
                if "disconnect" in err_msg or "not connected" in err_msg or "closed" in err_msg:
                    is_connected = False
                    break
                # If it's something else, try to continue if possible, or break
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
    finally:
        logger.info("📡 WebSocket loop terminated")


@app.websocket("/ws/stream/{source_type}")
async def server_stream_feed(websocket: WebSocket, source_type: str):
    """
    Server-side WebSocket (IP/USB/RTSP Camera Mode).
    Opens camera on server → Runs inference → Sends annotated frames.

    Model is warmed in background on startup; do not load synchronously here.
    """
    await websocket.accept()

    url_param = websocket.query_params.get('url')
    device_param = websocket.query_params.get('device')

    logger.info(f"📡 Server stream connected: type={source_type}, url={url_param}")

    # Handshake
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "connected",
        "message": f"Connected to {source_type} stream"
    }))

    # Resolve camera source
    camera_source = None
    if source_type == 'usb':
        camera_source = int(device_param) if device_param else 0
    elif source_type == 'ip':
        if url_param:
            camera_source = url_param
        else:
            await websocket.send_text(json.dumps({"type": "error", "error": "IP camera URL required"}))
            return
    elif source_type == 'rtsp':
        if url_param:
            camera_source = url_param
        else:
            await websocket.send_text(json.dumps({"type": "error", "error": "RTSP stream URL required"}))
            return

    # Fallback to local camera
    if camera_source is None:
        camera_source = 0

    import cv2
    import numpy as np

    camera = None
    frame_count = 0
    detector = None

    try:
        # Open camera
        if camera_source is not None:
            camera = cv2.VideoCapture(camera_source)
            if isinstance(camera_source, str):
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not camera.isOpened():
                logger.error(f"❌ Could not open {source_type}: {camera_source}")
                await websocket.send_text(json.dumps({"type": "error", "error": f"Failed to connect to {source_type} camera"}))
                # Keep connection open but idle
                while True:
                    await asyncio.sleep(1)
                return

            logger.info(f"✅ {source_type.upper()} camera opened")

        # Stream loop
        while True:
            fire_detected = False
            confidence = 0.0
            detection_boxes = []
            frame = None

            if camera and camera.isOpened():
                success, frame = camera.read()
                if not success:
                    logger.warning(f"⚠️  Failed to read frame from {source_type}")
                    break

                # Ensure model is loaded (fallback to synchronous loading if warm-up didn't complete)
                if detector is None:
                    logger.info("⏳ Loading model for server stream...")
                    detector = get_model()
                    if not detector:
                        logger.error("❌ Failed to acquire model")
                        break
                    state.model_loaded = True

                # 1. FAST YOLO INFERENCE
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res_obj = await asyncio.to_thread(
                    process_inference,
                    rgb_frame,
                    threshold=state.confidence_threshold,
                    lat=lat, lng=lng,
                    force_ai=False,
                    location_name=location_name,
                    skip_ai=True
                )
                
                fire_detected = res_obj.fire_detected
                confidence = res_obj.confidence
                detection_boxes = res_obj.detections

                # 2. TRIGGER BACKGROUND AI
                periodic_check = (frame_count % 50 == 0) # Less frequent for server stream
                current_time = time.time()
                can_call_ai = (current_time - _last_ai_time) >= AI_COOLDOWN
                
                if (res_obj.yolo_suspect or periodic_check) and can_call_ai:
                    asyncio.create_task(background_ai_verification(
                        websocket, rgb_frame, lat, lng, location_name
                    ))
            else:
                # No camera: send placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"NO SOURCE FOR {source_type.upper()}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Check camera configuration", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                await asyncio.sleep(0.5)

            # Encode frame as base64 JPEG
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_data = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logger.error(f"Frame encoding error: {e}")
                continue

            # Send payload
            payload = {
                "type": "frame",
                "data": image_data,
                "timestamp": datetime.now().isoformat(),
                "frame_id": frame_count,
                "detection": {
                    "fire_detected": fire_detected,
                    "confidence": float(confidence),
                    "fire_type": "Fire" if fire_detected else None,
                    "boxes": detection_boxes
                }
            }

            try:
                await websocket.send_text(json.dumps(payload))
                frame_count += 1
            except WebSocketDisconnect:
                logger.info(f"📡 Stream {source_type} client disconnected")
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info(f"📡 Stream {source_type} disconnected")
                    break
                logger.error(f"❌ Send error: {e}")
                break
            except Exception as e:
                logger.error(f"❌ Send error: {e}")
                break

            # Control frame rate (~30 FPS)
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        logger.info(f"📡 Stream {source_type} disconnected")
    except Exception as e:
        logger.error(f"❌ Stream error: {e}")
    finally:
        if camera:
            camera.release()
        logger.info(f"📡 Stream {source_type} closed")



# ============================================================================
# NO app.run() or if __name__ == "__main__" block
# Railway will run: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
# ============================================================================

logger.info("🚀 Fire Detection API ready for Railway deployment")
