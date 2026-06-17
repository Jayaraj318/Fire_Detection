from openai import OpenAI
import PIL.Image
import os
import json
import logging
import base64
from io import BytesIO
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class GPTFireDetector:
    """
    Fire detection using OpenAI ChatGPT (GPT-4o).
    Acts as a secondary verification for Gemini.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize GPT detector.
        """
        self.client = OpenAI(api_key=api_key)
        self.model_id = "gpt-4o"
        logger.info(f"✅ GPT Fire Detector initialized with {self.model_id}")

    def _encode_image(self, image_np):
        """Convert numpy array to base64 for OpenAI."""
        img = PIL.Image.fromarray(image_np)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def detect(self, image_np) -> Dict:
        """
        Analyze image for fire/smoke using GPT-4o.
        """
        try:
            base64_image = self._encode_image(image_np)
            
            prompt = """
            Analyze this image for fire or smoke. 
            PAY SPECIAL ATTENTION to small, bright light sources like matchsticks, candles, or lighters.
            In dark environments, a matchstick flame will appear as a very bright, concentrated spot.
            
            Return ONLY a valid JSON object with the following structure:
            {
                "fire_detected": boolean,
                "confidence": float (0.0 to 1.0),
                "fire_type": string ("Matchstick", "Candle", "Lighter", "Large Fire", "Smoke", "None"),
                "description": string (brief explanation),
                "bboxes": list of list of floats [[ymin, xmin, ymax, xmax]] normalized 0 to 1
            }
            """
            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low" # Use low detail to save tokens and stay within rate limits
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            logger.info(f"GPT [{self.model_id}]: {result['fire_detected']} ({result.get('confidence', 0.0):.2%}) - {result.get('fire_type')}")
            return result
            
        except Exception as e:
            if "429" in str(e):
                logger.warning("⚠️ GPT rate limit exceeded (429).")
            else:
                logger.error(f"❌ GPT Inference failed: {e}")
            
            return {
                "fire_detected": False,
                "confidence": 0.0,
                "fire_type": "None",
                "description": f"OpenAI Error: {str(e)}"
            }
