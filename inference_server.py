"""
Flask API for MNIST Digit Recognition
Serves predictions from the trained SimpleNN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms

# Define the model architecture (must match training)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Load model
device = torch.device('cpu')  # Use CPU for simplicity
model = SimpleNN().to(device)
model.load_state_dict(torch.load('models/mnist_model.pth', map_location=device))
model.eval()

print("✓ Model loaded successfully!")
print("✓ Server ready to accept requests")

# Image preprocessing transform (must match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess_image(image_data):
    """
    Preprocess base64 image data for model input.
    
    Args:
        image_data: Base64 encoded image string
    
    Returns:
        Preprocessed tensor ready for model
    """
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Invert colors: canvas is black-on-white, MNIST is white-on-black
    image = ImageOps.invert(image)
    
    # Apply transforms
    tensor = transform(image)
    
    return tensor


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict digit from image.
    
    Expected JSON:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns JSON:
    {
        "digit": 7,
        "confidence": 0.95,
        "probabilities": [0.01, 0.02, ..., 0.95, ...]
    }
    """
    try:
        # Get image from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        image_tensor = preprocess_image(data['image'])
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(device)
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[prediction].item()
        
        # Return results
        return jsonify({
            'digit': prediction,
            'confidence': float(confidence),
            'probabilities': probabilities.cpu().numpy().tolist()
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MNIST Digit Recognition API Server")
    print("="*60)
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  POST /predict - Predict digit from image")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
