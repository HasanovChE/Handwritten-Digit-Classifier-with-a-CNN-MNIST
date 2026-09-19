# api/app_onnx.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

app = FastAPI(title="MNIST ONNX Digit Classifier API")

ORT_SESSION = ort.InferenceSession("models/tensorflow_cnn.onnx")
INPUT_NAME = ORT_SESSION.get_inputs()[0].name

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.get("/")
def root():
    return {"message": "MNIST ONNX Runtime Digit Classifier"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    processed = preprocess_image(image)
    
    outputs = ORT_SESSION.run(None, {INPUT_NAME: processed})
    probabilities = outputs[0][0]
    
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    return {
        "digit": digit,
        "confidence": confidence
    }