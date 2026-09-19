# convert_to_onnx.py
import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model("models/tensorflow_cnn.keras")

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name="input")])
def inference_func(input_tensor):
    return model(input_tensor, training=False)

onnx_model, _ = tf2onnx.convert.from_function(
    inference_func,
    input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name="input")],
    opset=13
)

onnx.save(onnx_model, "tensorflow_cnn.onnx")
print("Model successfully converted and saved as 'tensorflow_cnn.onnx'!")

