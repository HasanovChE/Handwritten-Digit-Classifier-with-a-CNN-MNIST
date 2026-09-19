# tests/test_api.py
import io
from fastapi.testclient import TestClient
from PIL import Image
from api.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "MNIST CNN Digit Classifier"}

def test_predict_endpoint():
    img = Image.new("L", (28, 28), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_digit.png", img_byte_arr, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "digit" in data
    assert "confidence" in data
    assert isinstance(data["digit"], int)
    assert isinstance(data["confidence"], float)
    assert 0 <= data["digit"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0