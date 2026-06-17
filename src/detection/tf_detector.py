import tensorflow as tf
import numpy as np
import cv2
import logging
import os

logger = logging.getLogger(__name__)

class TFDetector:
    def __init__(self, model_path, confidence_threshold=0.10):
        self.model_path = model_path
        self.conf_thres = confidence_threshold
        self.model = None
        self.input_shape = (224, 224)  # Default, will try to read from model
        self.load_model()
        
    def load_model(self):
        try:
            logger.info(f"Loading .h5 model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Try to infer input shape
            if self.model.input_shape and len(self.model.input_shape) == 4:
                h = self.model.input_shape[1]
                w = self.model.input_shape[2]
                if h and w:
                    self.input_shape = (h, w)
            logger.info(f"Model loaded. Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def color_based_fire_check(self, frame):
        """
        Supplementary color-based fire/flame detection for small flames
        like matchsticks that the neural network might miss.
        Returns list of [x1, y1, x2, y2, confidence, class_id] detections.
        """
        detections = []
        
        # Convert to HSV for fire color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire/flame color ranges in HSV
        # Range 1: Red-orange flames (low hue)
        lower_fire1 = np.array([0, 80, 200])
        upper_fire1 = np.array([25, 255, 255])
        
        # Range 2: Red flames (high hue, wrapping around)
        lower_fire2 = np.array([160, 80, 200])
        upper_fire2 = np.array([180, 255, 255])
        
        # Range 3: Yellow-white hot flames (very bright)
        lower_fire3 = np.array([20, 30, 220])
        upper_fire3 = np.array([45, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask3 = cv2.inRange(hsv, lower_fire3, upper_fire3)
        
        fire_mask = mask1 | mask2 | mask3
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.0001  # Minimum 0.01% of frame area (tiny sparks/matchsticks)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Fire-like aspect ratio check (flames are usually taller than wide, or roughly square)
            aspect = bh / max(bw, 1)
            if aspect < 0.3 or aspect > 5.0:
                continue
            
            # Calculate confidence based on area coverage and brightness
            roi = frame[y:y+bh, x:x+bw]
            brightness = np.mean(roi)
            area_ratio = area / (bw * bh) if (bw * bh) > 0 else 0
            
            # Higher brightness and area ratio = higher confidence
            conf = min(0.85, 0.30 + (brightness / 255.0) * 0.4 + area_ratio * 0.2)
            
            if conf >= 0.25:
                logger.info(f"🔥 Color-based fire detected! Conf: {conf:.2f} Area: {area}")
                detections.append([float(x), float(y), float(x + bw), float(y + bh), float(conf), 0])
        
        return detections

    def process_output(self, outputs, img_shape):
        """
        Decode YOLO outputs.
        3 outputs: 7x7, 14x14, 28x28 (stride 32, 16, 8)
        Each output shape: (1, grid, grid, 18)
        18 channels = 3 anchors * (1 class + 5 params: x, y, w, h, obj_conf)
        """
        boxes = []
        confidences = []
        class_ids = []
        
        h, w = self.input_shape
        
        # Standard YOLO anchors
        anchors = {
            8:  [(10, 13), (16, 30), (33, 23)],
            16: [(30, 61), (62, 45), (59, 119)],
            32: [(116, 90), (156, 198), (373, 326)]
        }
        
        # Map outputs by grid size to correct strides
        layer_outputs = []
        for out in outputs:
            grid_h, grid_w = out.shape[1], out.shape[2]
            stride = h // grid_h
            layer_outputs.append((out, stride))
            
        for output, stride in layer_outputs:
            grid_h, grid_w = output.shape[1], output.shape[2]
            pred = output.reshape((1, grid_h, grid_w, 3, 6))

            # Grid coordinates
            grid_x = np.arange(grid_w, dtype=np.float32)
            grid_y = np.arange(grid_h, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)
            
            # Decode box center
            box_xy = self.sigmoid(pred[..., 0:2])
            pred_xy = (box_xy + np.stack((grid_x, grid_y), axis=-1)) * stride

            # Calculate scores
            obj_conf = self.sigmoid(pred[..., 4:5])
            class_conf = self.sigmoid(pred[..., 5:])
            scores = obj_conf * class_conf

            # Decode box dimensions
            box_wh = np.exp(np.clip(pred[..., 2:4], -10, 10))
            anchor_grid = np.array(anchors.get(stride, [(1,1)]*3)).reshape((1, 1, 1, 3, 2))
            pred_wh = box_wh * anchor_grid
            
            # Convert to x1, y1, x2, y2
            pred_x1y1 = pred_xy - pred_wh / 2
            pred_x2y2 = pred_xy + pred_wh / 2
            
            # Use a very low threshold to catch everything, filter later
            detection_threshold = max(self.conf_thres * 0.5, 0.02)
            indices = np.where(scores > detection_threshold)
            
            if len(indices[0]) > 0:
                for i in range(len(indices[0])):
                    bi, yi, xi, ai, ci = indices[0][i], indices[1][i], indices[2][i], indices[3][i], indices[4][i]
                    score = scores[bi, yi, xi, ai, ci]
                    
                    x1 = pred_x1y1[bi, yi, xi, ai, 0]
                    y1 = pred_x1y1[bi, yi, xi, ai, 1]
                    x2 = pred_x2y2[bi, yi, xi, ai, 0]
                    y2 = pred_x2y2[bi, yi, xi, ai, 1]
                    
                    # Clamp to valid range
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(score))
                        class_ids.append(int(ci))
                    
        return boxes, confidences, class_ids

    def predict(self, frame):
        """
        Run inference on a BGR frame from OpenCV.
        The .h5 model expects BGR 0-255 input (confirmed by diagnostic testing).
        """
        # Resize to model input (keep BGR, keep 0-255 range)
        input_img = cv2.resize(frame, self.input_shape)
        input_tensor = input_img.astype(np.float32) 
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Run model
        outputs = self.model.predict(input_tensor, verbose=0)
        
        # Log objectness for debugging (every call, but keep it brief)
        best_obj = 0.0
        best_score = 0.0
        for out in outputs:
            grid_h, grid_w = out.shape[1], out.shape[2]
            pred = out.reshape((1, grid_h, grid_w, 3, 6))
            obj_conf = self.sigmoid(pred[..., 4])
            cls_conf = self.sigmoid(pred[..., 5])
            score = obj_conf * cls_conf
            best_obj = max(best_obj, obj_conf.max())
            best_score = max(best_score, score.max())
        
        logger.info(f"H5 Model: best_obj={best_obj:.4f} best_score={best_score:.4f} threshold={self.conf_thres}")
        
        # Process YOLO output
        boxes, confs, classes = self.process_output(outputs, frame.shape)
        
        # Also run color-based detection as supplementary
        color_detections = self.color_based_fire_check(frame)
        
        # Merge neural network detections with color-based detections
        all_boxes = list(boxes)
        all_confs = list(confs)
        all_classes = list(classes)
        
        for det in color_detections:
            x1, y1, x2, y2, conf, cls_id = det
            # Scale color detections to model coordinate space for NMS
            orig_h, orig_w = frame.shape[:2]
            model_h, model_w = self.input_shape
            sx = model_w / orig_w
            sy = model_h / orig_h
            all_boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
            all_confs.append(conf)
            all_classes.append(int(cls_id))
        
        logger.info(f"Total detections: {len(all_boxes)} (NN: {len(boxes)}, Color: {len(color_detections)})")
        
        # NMS
        if not all_boxes:
            return []
            
        # Convert boxes to [x, y, w, h] format for NMS
        nms_boxes = []
        for b in all_boxes:
            nms_boxes.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
        
        indices = cv2.dnn.NMSBoxes(nms_boxes, all_confs, self.conf_thres, 0.45)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = all_boxes[i]
                
                # Scale boxes to original frame
                orig_h, orig_w = frame.shape[:2]
                model_h, model_w = self.input_shape
                
                x1 = x1 * (orig_w / model_w)
                y1 = y1 * (orig_h / model_h)
                x2 = x2 * (orig_w / model_w)
                y2 = y2 * (orig_h / model_h)
                
                results.append([x1, y1, x2, y2, all_confs[i], all_classes[i]])
                
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    MODEL_PATH = r"c:\Users\User\Documents\fire\Fire_Detection\models\yolo\weights\detection_model-ex-33--loss-4.97.h5"
    if os.path.exists(MODEL_PATH):
        detector = TFDetector(MODEL_PATH)
        # Dummy frame
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.predict(dummy)
        print("Detections:", res)
