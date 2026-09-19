# validate_model_gate.py
import tensorflow as tf
import numpy as np

_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

model = tf.keras.models.load_model("tensorflow_cnn.keras")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Evaluated Model Accuracy: {accuracy:.4f}")

MIN_ACCURACY_THRESHOLD = 0.98

if accuracy < MIN_ACCURACY_THRESHOLD:
    print(f"❌ Quality Gate Failed! Accuracy ({accuracy:.4f}) is below the required threshold ({MIN_ACCURACY_THRESHOLD}).")
    exit(1) 
else:
    print(f"✅ Quality Gate Passed! Model meets production standards.")