# src/libs/classify_color.py

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "src/models/cnn_color.onnx"
INPUT_SIZE = 64  # Adjust if your model expects another size

def load_color_model():
    return ort.InferenceSession(MODEL_PATH)

def preprocess_color_image(img):
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, axis=0)

def classify_color(img, model):
    input_tensor = preprocess_color_image(img)
    ort_inputs = {model.get_inputs()[0].name: input_tensor}
    ort_outs = model.run(None, ort_inputs)
    
    logits = ort_outs[0][0]
    class_index = int(np.argmax(logits))
    
    return "white" if class_index == 1 else "black"
