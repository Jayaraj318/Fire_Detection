
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
from typing import Optional, List, Dict
import os

logger = logging.getLogger(__name__)

# Try to import Twilio
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not installed. SMS notifications will be disabled. Install with: pip install twilio")


class EmailNotifier:
    def __init__(self, config: dict):
        """
        Initialize the EmailNotifier.
        
        Args:
            config: Dictionary containing email configuration:
                - smtp_server
                - smtp_port
                - sender_email
                - sender_password
                - receiver_emails (list or comma-separated string)
                - cooldown_seconds (default: 300)
        """
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email', '')
        self.sender_password = config.get('sender_password', '')
        
        receivers = config.get('receiver_emails', [])
        if isinstance(receivers, str):
            self.receiver_emails = [r.strip() for r in receivers.split(',')]
        else:
            self.receiver_emails = receivers
            
        self.cooldown_seconds = config.get('cooldown_seconds', 300) # 5 minutes default
        self.last_alert_time = 0
        
        # Twilio config
        self.twilio_account_sid = config.get('twilio_account_sid', '')
        self.twilio_auth_token = config.get('twilio_auth_token', '')
        self.twilio_phone_number = config.get('twilio_phone_number', '')
        self.receiver_phone_numbers = config.get('receiver_phone_numbers', [])
        
        if isinstance(self.receiver_phone_numbers, str):
            self.receiver_phone_numbers = [p.strip() for p in self.receiver_phone_numbers.split(',')]
        
        # Initialize Twilio client if credentials available
        self.twilio_client = None
        if TWILIO_AVAILABLE and self.twilio_account_sid and self.twilio_auth_token:
            try:
                self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
                logger.info("✅ Twilio SMS client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Twilio client: {e}")
        elif not self.twilio_account_sid:
            logger.info("ℹ️ Twilio not configured. SMS alerts will be disabled.")
        
    def send_fire_alert(self, confidence: float, image_path: Optional[str] = None, location: str = "Unknown", lat: Optional[float] = None, lng: Optional[float] = None, address: Optional[str] = None):
        """
        Send fire alert via email and SMS.
        
        Args:
            confidence: Confidence score of the detection (0-1).
            image_path: Path to the image file showing the detection (optional).
            location: Description of the location (optional).
            lat: Latitude (optional).
            lng: Longitude (optional).
            address: Resolved address (optional).
        """
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (current_time - self.last_alert_time))
            logger.info(f"⏳ Alert suppressed due to cooldown. Next alert allowed in {remaining}s.")
            return

        loc_str = location
        if address:
            loc_str = f"{location} ({address})"
        elif lat and lng:
            loc_str = f"{location} (Lat: {lat}, Lng: {lng})"

        logger.info(f"🔥 Sending fire alert (confidence: {confidence:.2%}, location: {loc_str})")
        
        # Send email
        self._send_email_alert(confidence, image_path, location, lat, lng, address)
        
        # SMS disabled as per user request ("receive only the mail not sms")
        # if self.twilio_client and self.receiver_phone_numbers:
        #    self._send_sms_alert(confidence, location)
        
        self.last_alert_time = current_time

    def _send_email_alert(self, confidence: float, image_path: Optional[str] = None, location: str = "Unknown", lat: Optional[float] = None, lng: Optional[float] = None, address: Optional[str] = None):
        """Send fire alert via email."""
        if not self.sender_email or not self.receiver_emails:
            logger.warning("⚠️ Email configuration incomplete. Email alert not sent.")
            return

        subject = f"🔥 FIRE DETECTED! (Confidence: {confidence:.2%})"
        
        loc_details = f"Location: {location}\n"
        if lat and lng:
            loc_details += f"Maps Link: https://www.google.com/maps/search/?api=1&query={lat},{lng}\n"

        body = f"""
URGENT: Fire detected by the monitoring system.

Confidence: {confidence:.2%}
{loc_details}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Please verify immediately and take action if necessary.

---
This is an automated alert from the Fire Detection System.
        """

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
            except Exception as e:
                logger.error(f"Failed to attach image: {e}")

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Fire alert email sent to {self.receiver_emails}")
        except smtplib.SMTPAuthenticationError:
            logger.error("❌ SMTP Authentication failed. Check email/password and use app-specific password for Gmail.")
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP error while sending email: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")

    def _send_sms_alert(self, confidence: float, location: str = "Unknown"):
        """Send fire alert via SMS using Twilio."""
        if not self.twilio_client or not self.receiver_phone_numbers:
            logger.warning("⚠️ Twilio configuration incomplete. SMS alert not sent.")
            return

        message_body = f"🔥 FIRE ALERT! Confidence: {confidence:.0%} at {location}. Time: {time.strftime('%H:%M:%S')}"

        for phone_number in self.receiver_phone_numbers:
            try:
                message = self.twilio_client.messages.create(
                    body=message_body,
                    from_=self.twilio_phone_number,
                    to=phone_number
                )
                logger.info(f"✅ Fire alert SMS sent to {phone_number} (SID: {message.sid})")
            except Exception as e:
                logger.error(f"❌ Failed to send SMS to {phone_number}: {e}")

    def get_status(self) -> Dict[str, bool]:
        """Get notification system status."""
        return {
            "email_configured": bool(self.sender_email and self.receiver_emails),
            "sms_configured": bool(self.twilio_client and self.receiver_phone_numbers),
            "email_recipients": len(self.receiver_emails),
            "sms_recipients": len(self.receiver_phone_numbers) if self.twilio_client else 0
        }
