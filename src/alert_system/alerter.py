"""
Fire Alert System
Multi-channel alert system using Twilio SMS and SMTP Email
"""

import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class FireAlertSystem:
    """
    IoT-based alert system for fire detection.
    
    Channels:
    - SMS via Twilio (free tier available)
    - Email via SMTP (Gmail, Outlook, etc.)
    
    Features:
    - Cooldown to prevent alert spam
    - Evidence image attachment with bounding boxes
    - Alert logging
    """
    
    def __init__(self, config_path: str = 'configs/alert_config.json'):
        """
        Initialize the alert system.
        
        Args:
            config_path: Path to alert configuration JSON file.
        """
        self.config = self._load_config(config_path)
        self.alert_log: List[Dict] = []
        self.last_alert_time: Optional[datetime] = None
        self.cooldown_seconds = self.config.get('cooldown_seconds', 60)
        
        # Ensure alerts directory exists
        Path('alerts').mkdir(exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default config
            return {
                'camera_location': 'Unknown Location',
                'cooldown_seconds': 60,
                'email': {'enabled': False},
                'sms': {'enabled': False}
            }
    
    def should_send_alert(self) -> bool:
        """Check if enough time has passed since last alert."""
        if self.last_alert_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        return elapsed > self.cooldown_seconds
    
    def send_alert(
        self,
        detection_result: Dict,
        frame: np.ndarray,
        heatmap: Optional[np.ndarray] = None
    ) -> bool:
        """
        Send multi-channel alert.
        
        Args:
            detection_result: Model output dictionary
            frame: Original frame [H, W, 3] (BGR)
            heatmap: Optional Grad-CAM heatmap [H, W]
            
        Returns:
            True if alert was sent, False if in cooldown
        """
        if not self.should_send_alert():
            print(f"⏳ Alert cooldown active ({self.cooldown_seconds}s)")
            return False
        
        # Prepare alert data
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'confidence': float(detection_result.get('confidence', [[0]])[0][0]) if hasattr(detection_result.get('confidence', 0), '__iter__') else float(detection_result.get('confidence', 0)),
            'fire_type': self._get_fire_type(detection_result.get('fire_type_probs')),
            'location': self.config.get('camera_location', 'Unknown'),
            'bounding_boxes': self._convert_boxes(detection_result.get('bounding_boxes', []))
        }
        
        # Save evidence image
        evidence_path = f"alerts/{alert_data['timestamp'].replace(':', '-')}.jpg"
        self._save_evidence(frame, detection_result.get('bounding_boxes', []), heatmap, evidence_path)
        alert_data['evidence_path'] = evidence_path
        
        # Send via enabled channels
        success = False
        
        if self.config.get('email', {}).get('enabled', False):
            try:
                self._send_email(alert_data, evidence_path)
                success = True
            except Exception as e:
                print(f"❌ Email failed: {e}")
        
        if self.config.get('sms', {}).get('enabled', False):
            try:
                self._send_sms(alert_data)
                success = True
            except Exception as e:
                print(f"❌ SMS failed: {e}")
        
        # Log alert
        self.alert_log.append(alert_data)
        self.last_alert_time = datetime.now()
        
        print(f"🔥 ALERT SENT: {alert_data['fire_type']} fire detected at {alert_data['location']}")
        print(f"   Confidence: {alert_data['confidence']:.1%}")
        
        return success
    
    def _get_fire_type(self, probs) -> str:
        """Convert fire type probabilities to class name."""
        if probs is None:
            return "Unknown Type"
        
        classes = ['Class A (Combustibles)', 'Class B (Flammable Liquids)', 'Class C (Electrical)']
        
        try:
            if hasattr(probs, 'argmax'):
                idx = probs.argmax().item() if hasattr(probs.argmax(), 'item') else probs.argmax()
            else:
                idx = 0
            return classes[idx]
        except:
            return "Unknown Type"
    
    def _convert_boxes(self, boxes) -> List:
        """Convert bounding boxes to list format."""
        if boxes is None or (hasattr(boxes, '__len__') and len(boxes) == 0):
            return []
        try:
            return boxes.tolist() if hasattr(boxes, 'tolist') else list(boxes)
        except:
            return []
    
    def _save_evidence(
        self,
        frame: np.ndarray,
        boxes,
        heatmap: Optional[np.ndarray],
        path: str
    ):
        """Save annotated frame with bounding boxes and heatmap."""
        frame_annotated = frame.copy()
        
        # Draw bounding boxes
        if boxes is not None and hasattr(boxes, '__len__') and len(boxes) > 0:
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame_annotated, 'FIRE', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except:
                    pass
        
        # Overlay heatmap if available
        if heatmap is not None:
            try:
                heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_resized * 255).astype('uint8'),
                    cv2.COLORMAP_JET
                )
                frame_annotated = cv2.addWeighted(frame_annotated, 0.7, heatmap_colored, 0.3, 0)
            except:
                pass
        
        cv2.imwrite(path, frame_annotated)
    
    def _send_email(self, alert_data: Dict, image_path: str):
        """Send email alert with evidence image via SMTP."""
        email_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from_address']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = f"🔥 FIRE ALERT - {alert_data['location']}"
        
        # Email body
        body = f"""
============================================
         FIRE DETECTION ALERT
============================================

📍 Location: {alert_data['location']}
🕐 Time: {alert_data['timestamp']}
📊 Confidence: {alert_data['confidence']:.1%}
🔥 Fire Type: {alert_data['fire_type']}

⚠️ RECOMMENDED ACTION:
Use appropriate fire extinguisher based on fire type.

Evidence image attached.

---
This is an automated alert from the Fire Detection System.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach evidence image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename='fire_evidence.jpg')
                msg.attach(img)
        
        # Send email
        try:
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            print("✅ Email alert sent successfully")
        except Exception as e:
            raise Exception(f"SMTP error: {e}")
    
    def _send_sms(self, alert_data: Dict):
        """Send SMS alert via Twilio."""
        try:
            from twilio.rest import Client
        except ImportError:
            print("⚠️ Twilio not installed. Run: pip install twilio")
            return
        
        sms_config = self.config['sms']
        
        try:
            client = Client(
                sms_config['twilio_account_sid'],
                sms_config['twilio_auth_token']
            )
            
            message_body = (
                f"🔥 FIRE ALERT!\n"
                f"Location: {alert_data['location']}\n"
                f"Confidence: {alert_data['confidence']:.0%}\n"
                f"Type: {alert_data['fire_type']}\n"
                f"Time: {alert_data['timestamp']}"
            )
            
            for to_number in sms_config.get('to_numbers', []):
                message = client.messages.create(
                    body=message_body,
                    from_=sms_config['from_number'],
                    to=to_number
                )
                print(f"✅ SMS sent to {to_number}: {message.sid}")
                
        except Exception as e:
            raise Exception(f"Twilio error: {e}")
    
    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alert history."""
        return self.alert_log[-limit:]
    
    def clear_cooldown(self):
        """Manually clear cooldown (for testing)."""
        self.last_alert_time = None
    
    def update_config(self, new_config: Dict):
        """Update configuration at runtime."""
        self.config.update(new_config)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 60)


class AlertLogger:
    """
    Persistent alert logging to file.
    """
    
    def __init__(self, log_dir: str = 'alerts/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
    
    def log_alert(self, alert_data: Dict):
        """Append alert to daily log file."""
        logs = []
        
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        
        logs.append(alert_data)
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_daily_alerts(self, date: Optional[str] = None) -> List[Dict]:
        """Get all alerts for a specific date."""
        if date is None:
            log_file = self.log_file
        else:
            log_file = self.log_dir / f"alerts_{date}.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return []


if __name__ == '__main__':
    # Test the alert system
    print("Fire Alert System Test")
    
    alerter = FireAlertSystem()
    
    # Create dummy detection result
    detection_result = {
        'confidence': [[0.92]],
        'fire_type_probs': None,
        'bounding_boxes': []
    }
    
    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, 'TEST FRAME', (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test alert (will save evidence but not send without config)
    alerter.send_alert(detection_result, dummy_frame)
    
    print("\nAlert history:")
    for alert in alerter.get_alert_history():
        print(f"  - {alert['timestamp']}: {alert['confidence']:.1%} confidence")
