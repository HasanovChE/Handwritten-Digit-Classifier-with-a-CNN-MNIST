# inference/predict_pytorch.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict_with_pytorch(image_path: str, model_path: str = "models/pytorch_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image = ImageOps.invert(image) 
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, dim=0)

    return predicted_class.item(), confidence.item()

if __name__ == "__main__":
    digit, conf = predict_with_pytorch("custom_images/test_digit.png")
    print(f"[PyTorch] Prediction: {digit}, Confidence: {conf:.4f}")