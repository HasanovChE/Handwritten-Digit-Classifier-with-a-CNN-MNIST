# retraining_pipeline.py

import tensorflow as tf
import numpy as np
import os


def retrain_model():
    print("🔄 Starting automated model retraining due to data drift or schedule...")

    # Model path
    model_path = os.getenv(
        "MODEL_PATH",
        "models/tensorflow_cnn.keras"
    )

    # Make sure the directory exists
    model_dir = os.path.dirname(model_path)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print(f"📦 Model will be saved to: {model_path}")

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=128,
        verbose=1
    )

    # Save model
    model.save(model_path)

    # Verify model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model was not created: {model_path}"
        )

    if os.path.getsize(model_path) == 0:
        raise RuntimeError(
            f"Model file is empty: {model_path}"
        )

    print("✅ Retraining successfully completed!")
    print(f"📦 Updated model artifact saved to: {model_path}")
    print(f"📏 Model size: {os.path.getsize(model_path)} bytes")


if __name__ == "__main__":
    retrain_model()
