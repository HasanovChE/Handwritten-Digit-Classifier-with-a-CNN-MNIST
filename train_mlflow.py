# train_mlflow.py
import mlflow
mlflow.tensorflow.autolog()

import tensorflow as tf
from tensorflow.keras import layers, Sequential

mlflow.set_experiment("mnist_cnn_experiment")

with mlflow.start_run():
    epochs = 5
    batch_size = 128
    
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    model = Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train_tf, y_train,
        validation_data=(x_val_tf, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    print("MLflow tracking completed successfully!")