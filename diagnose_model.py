import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = r"c:\Users\User\Documents\fire\Fire_Detection\models\yolo\weights\detection_model-ex-33--loss-4.97.h5"

# Write results to a file to avoid encoding issues
with open("diagnose_results.txt", "w") as f:
    model = tf.keras.models.load_model(MODEL_PATH)
    f.write(f"INPUT SHAPE: {model.input_shape}\n")
    f.write(f"NUM OUTPUTS: {len(model.outputs)}\n")
    for i, o in enumerate(model.outputs):
        f.write(f"OUTPUT {i} SHAPE: {o.shape}\n")
    
    # Test with normalized 0-1 input
    h = model.input_shape[1] or 224
    w = model.input_shape[2] or 224
    dummy = np.random.rand(1, h, w, 3).astype(np.float32)
    results = model.predict(dummy, verbose=0)
    if isinstance(results, list):
        for i, r in enumerate(results):
            f.write(f"PRED_NORM {i} shape={r.shape} min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f}\n")
    else:
        f.write(f"PRED_NORM shape={results.shape} min={results.min():.4f} max={results.max():.4f}\n")
    
    # Test with 0-255 input (how tf_detector currently sends it)
    dummy255 = (np.random.rand(1, h, w, 3) * 255.0).astype(np.float32)
    results2 = model.predict(dummy255, verbose=0)
    if isinstance(results2, list):
        for i, r in enumerate(results2):
            f.write(f"PRED_255 {i} shape={r.shape} min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f}\n")
    else:
        f.write(f"PRED_255 shape={results2.shape} min={results2.min():.4f} max={results2.max():.4f}\n")
    
    # Check what sigmoid of objectness channel looks like
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    f.write("\n=== OBJECTNESS ANALYSIS (normalized input) ===\n")
    for i, r in enumerate(results if isinstance(results, list) else [results]):
        grid_h, grid_w = r.shape[1], r.shape[2]
        if r.shape[3] == 18:
            pred = r.reshape((1, grid_h, grid_w, 3, 6))
            obj_conf = sigmoid(pred[..., 4])
            cls_conf = sigmoid(pred[..., 5])
            f.write(f"Out {i} ({grid_h}x{grid_w}): obj_max={obj_conf.max():.4f} cls_max={cls_conf.max():.4f}\n")
        elif r.shape[3] == 255:
            f.write(f"Out {i} ({grid_h}x{grid_w}): COCO-style 85*3=255 channels\n")
        else:
            f.write(f"Out {i} ({grid_h}x{grid_w}): {r.shape[3]} channels\n")
    
    f.write("\n=== OBJECTNESS ANALYSIS (0-255 input) ===\n")
    for i, r in enumerate(results2 if isinstance(results2, list) else [results2]):
        grid_h, grid_w = r.shape[1], r.shape[2]
        if r.shape[3] == 18:
            pred = r.reshape((1, grid_h, grid_w, 3, 6))
            obj_conf = sigmoid(pred[..., 4])
            cls_conf = sigmoid(pred[..., 5])
            f.write(f"Out {i} ({grid_h}x{grid_w}): obj_max={obj_conf.max():.4f} cls_max={cls_conf.max():.4f}\n")
        else:
            f.write(f"Out {i} ({grid_h}x{grid_w}): {r.shape[3]} channels\n")

    # Test with a real fire image if debug_frame exists
    debug_frame = "debug_frame_0.jpg"
    if os.path.exists(debug_frame):
        f.write(f"\n=== REAL FRAME TEST ({debug_frame}) ===\n")
        img = cv2.imread(debug_frame)
        f.write(f"Frame shape: {img.shape}\n")
        
        # Test with RGB normalized
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, 0)
        
        results_real = model.predict(img_batch, verbose=0)
        if isinstance(results_real, list):
            for i, r in enumerate(results_real):
                grid_h, grid_w = r.shape[1], r.shape[2]
                if r.shape[3] == 18:
                    pred = r.reshape((1, grid_h, grid_w, 3, 6))
                    obj_conf = sigmoid(pred[..., 4])
                    cls_conf = sigmoid(pred[..., 5])
                    score = obj_conf * cls_conf
                    f.write(f"REAL_RGB_NORM Out {i} ({grid_h}x{grid_w}): obj_max={obj_conf.max():.4f} cls_max={cls_conf.max():.4f} score_max={score.max():.4f}\n")
        
        # Test with BGR 0-255 (how tf_detector currently does it)
        img_bgr = cv2.resize(img, (w, h))
        img_bgr_batch = np.expand_dims(img_bgr.astype(np.float32), 0)
        results_bgr = model.predict(img_bgr_batch, verbose=0)
        if isinstance(results_bgr, list):
            for i, r in enumerate(results_bgr):
                grid_h, grid_w = r.shape[1], r.shape[2]
                if r.shape[3] == 18:
                    pred = r.reshape((1, grid_h, grid_w, 3, 6))
                    obj_conf = sigmoid(pred[..., 4])
                    cls_conf = sigmoid(pred[..., 5])
                    score = obj_conf * cls_conf
                    f.write(f"REAL_BGR_255 Out {i} ({grid_h}x{grid_w}): obj_max={obj_conf.max():.4f} cls_max={cls_conf.max():.4f} score_max={score.max():.4f}\n")

print("Results written to diagnose_results.txt")
