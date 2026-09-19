# locustfile.py
import io
from PIL import Image
from locust import HttpUser, task, between

class MNISTUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        img = Image.new("L", (28, 28), color=0)
        self.img_byte_arr = io.BytesIO()
        img.save(self.img_byte_arr, format="PNG")
        self.img_byte_arr.seek(0)

    @task
    def predict_digit(self):
        self.img_byte_arr.seek(0)
        files = {"file": ("test_digit.png", self.img_byte_arr, "image/png")}
        self.client.post("/predict", files=files)