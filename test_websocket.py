import asyncio
import websockets
import base64
import cv2
import numpy as np
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WS_TEST")

async def test_websocket():
    uri = "ws://localhost:8000/ws/video"
    # Create a dummy image (red square)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), -1)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    # Convert buffer to bytes
    img_bytes = buffer.tobytes()
    # Encode bytes to base64 string
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_b64}"
    
    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected!")
            
            # Send the frame
            logger.info(f"📤 Sending frame (len={len(data_url)})...")
            await websocket.send(data_url)
            logger.info("✅ Frame sent")
            
            # Wait for response
            response = await websocket.recv()
            logger.info(f"📥 Received response: {response}")
            
            data = json.loads(response)
            if data.get("fire_detected"):
                logger.info("🔥 FIRE DETECTED in test image!")
            else:
                logger.info("✅ No fire detected (expected for red square without fire features)")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")

if __name__ == "__main__":
    if hasattr(asyncio, 'run'):
        asyncio.run(test_websocket())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_websocket())
