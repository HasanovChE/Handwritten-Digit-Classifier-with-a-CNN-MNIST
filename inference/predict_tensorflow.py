# inference/predict_tensorflow.py
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

def predict_with_tensorflow(image_path: str, model_path: str = "models/tensorflow_cnn.keras"):
    model = tf.keras.models.load_model(model_path)

    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1) 
    input_tensor = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return predicted_class, confidence

if __name__ == "__main__":
    digit, conf = predict_with_tensorflow("custom_images/test_digit.png")
    print(f"[TensorFlow] Prediction: {digit}, Confidence: {conf:.4f}")