import sys
import os
print(f"Python Executable: {sys.executable}")
print(f"Sys Path: {sys.path}")
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = r"c:\Users\User\Documents\fire\Fire_Detection\models\yolo\weights\detection_model-ex-33--loss-4.97.h5"

def inspect():
    print(f"TensorFlow Version: {tf.__version__}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        return

    print(f"🔍 Loading model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        try:
            config = model.get_config()
            print("\n--- Model Config Keys ---")
            print(list(config.keys()))
            # deep print layers to find anchors?
            # print(config) 
        except:
            print("No config available.")
            
        print("\n--- Model Summary ---")
        model.summary()
        
        print("\n--- Input Details ---")
        input_shape = model.input_shape
        print(f"Input Shape: {input_shape}")
        
        print("\n--- Output Details ---")
        output_shape = model.output_shape
        print(f"Output Shape: {output_shape}")
        
        # Create dummy input
        # Assuming (1, H, W, 3) or (1, H, W, 1)
        h, w = 224, 224 # default guess
        if input_shape and len(input_shape) >= 3:
             h = input_shape[1] if input_shape[1] else 224
             w = input_shape[2] if input_shape[2] else 224
             
        print(f"\n🧪 Running dummy inference with shape ({h}, {w})...")
        dummy_input = np.zeros((1, h, w, 3), dtype=np.float32)
        
        prediction = model.predict(dummy_input)
        print(f"✅ Prediction successful!")
        print(f"Prediction type: {type(prediction)}")
        if isinstance(prediction, list):
            for i, p in enumerate(prediction):
                print(f"Output {i} shape: {p.shape}")
                print(f"Output {i} values (first 5): {p.flatten()[:5]}")
        else:
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction values (first 5): {prediction.flatten()[:5]}")

    except Exception as e:
        print(f"❌ Failed to inspect model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect()
