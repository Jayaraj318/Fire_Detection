from google import genai
from google.genai import types
import PIL.Image
import os
import json
import logging
from typing import Dict, Optional
import io

logger = logging.getLogger(__name__)

class GeminiFireDetector:
    """
    Fire detection using the new Google GenAI SDK.
    Capable of detecting small fires like matchsticks with Gemini 2.0 Flash.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini detector using the new google.genai Client.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"
        logger.info(f"✅ Gemini Fire Detector initialized with {self.model_id}")

    def detect(self, image_np) -> Dict:
        """
        Analyze image for fire/smoke using Gemini 2.0.
        """
        try:
            # Convert numpy array to PIL Image
            img = PIL.Image.fromarray(image_np)
            
            # Prepare improved prompt for small/bright fires
            prompt = """
            Analyze this image for fire or smoke. 
            PAY SPECIAL ATTENTION to small, bright light sources that could be flames (matchsticks, candles, lighters).
            In indoor or dark environments, a matchstick flame will appear as a very bright, concentrated white/yellow/orange spot.
            
            Look for:
            1. Sharp glowing cores (flames).
            2. Any associated smoke or heat haze.
            3. Contextual clues (hand holding a match, candle wick).

            Return ONLY a valid JSON object with the following structure:
            {
                "fire_detected": boolean,
                "confidence": float (0.0 to 1.0),
                "fire_type": string (e.g., "Matchstick", "Candle", "Lighter", "Large Fire", "Smoke", "None"),
                "description": string (brief explanation of what you see),
                "bboxes": list of list of floats [[ymin, xmin, ymax, xmax]] normalized 0 to 1
            }
            """
            
            # Use the new SDK generate_content
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                )
            )
            
            # Robustly parse the response
            text = response.text.strip()
            logger.debug(f"Gemini Raw Text: {text}")
            
            # Clean text if it has markdown formatting (though mime_type should prevent it)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            logger.info(f"Gemini [{self.model_id}]: {result['fire_detected']} ({result.get('confidence', 0.0):.2%}) - {result.get('fire_type')}")
            return result
            
        except Exception as e:
            if "429" in str(e):
                logger.warning("⚠️ Gemini rate limit exceeded (429). Skipping detection.")
            else:
                logger.error(f"❌ Gemini Inference failed: {e}")
            
            return {
                "fire_detected": False,
                "confidence": 0.0,
                "fire_type": "None",
                "description": f"SDK Error: {str(e)}"
            }
