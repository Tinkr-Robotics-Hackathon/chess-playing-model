# src/detect_board.py

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "src/models/yolo_corner.onnx"
INPUT_SIZE = 640  # standard YOLO size

def load_model():
    return ort.InferenceSession(MODEL_PATH)

def preprocess(image):
    original_shape = image.shape[:2]  # (H, W)
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), original_shape


def postprocess(outputs, original_shape):
    pred = outputs[0]
    pred = pred.T  # YOLO output
    boxes = []
    for det in pred:
        conf = float(det[4])
        if conf < 0.3:
            continue
        x, y, w, h = det[0:4]

        # These are normalized wrt 640x640, so map back to original image
        x *= original_shape[1] / INPUT_SIZE
        y *= original_shape[0] / INPUT_SIZE
        w *= original_shape[1] / INPUT_SIZE
        h *= original_shape[0] / INPUT_SIZE

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        boxes.append((cx, cy))
    return order_points(boxes)


def order_points(points):
    """ Orders 4 points as top-left, top-right, bottom-right, bottom-left """
    points = np.array(points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    
    # divide all the points by 100 so they are in scale
    # points = points / 1000.0

    return np.array([
        points[np.argmin(s)],    # top-left
        points[np.argmin(diff)], # top-right
        points[np.argmax(s)],    # bottom-right
        points[np.argmax(diff)], # bottom-left
    ])

def detect_corners(image, model):
    inp, shape = preprocess(image)
    ort_inputs = {model.get_inputs()[0].name: inp}
    ort_outs = model.run(None, ort_inputs)
    corners = postprocess(ort_outs, shape)
    return corners
