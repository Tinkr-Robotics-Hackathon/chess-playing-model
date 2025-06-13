# src/libs/classify_piece.py

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "src/models/cnn_piece.onnx"
INPUT_SIZE = 64  # adapt if model expects a different size

def load_piece_model():
    return ort.InferenceSession(MODEL_PATH)

def preprocess_piece_image(img):
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, axis=0)

def classify_piece(img, model):
    input_tensor = preprocess_piece_image(img)
    ort_inputs = {model.get_inputs()[0].name: input_tensor}
    ort_outs = model.run(None, ort_inputs)
    
    # Output is like [[-10.05, 9.87]] — apply softmax or just use argmax
    logits = ort_outs[0][0]
    class_index = int(np.argmax(logits))
    
    return "piece" if class_index == 1 else "empty"
