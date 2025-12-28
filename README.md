# AI Tutorial: Handwritten Digit Recognition

A complete guide to training a neural network for handwritten digit recognition, from Jupyter notebooks to a real-world React Native mobile application.

## 📁 Project Overview

This repository is split into two main parts:
1.  **AI Exercises**: Jupyter notebooks teaching PyTorch, MNIST, and Vision Transformers.
2.  **DigitRecognizer**: A React Native app for Samsung Galaxy S24 Ultra (and other Android/iOS devices) that recognizes digits drawn with a stylus or finger.

---

## 🚀 Quick Start (Python & AI)

We use **[uv](https://astral.sh/uv/)** for fast, reliable Python dependency management.

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup & Run Notebooks
```bash
# Clone the repository
git clone https://github.com/ztrangercode/ai-tutorial.git
cd ai-tutorial

# Install dependencies and start Jupyter
uv run jupyter notebook
```

## 📚 Exercises

1.  **01_hello_pytorch.ipynb**: Introduction to Tensors and PyTorch basics.
2.  **02_load_mnist.ipynb**: Loading and exploring the MNIST dataset.
3.  **03_create_model.ipynb**: Building a neural network for digit recognition.
4.  **04_train_model.ipynb**: Training the neural network.
5.  **05_evaluate_model.ipynb**: Evaluating the model's performance.
6.  **06_inference.ipynb**: Using the trained model for inference on new images.
7.  **07_convert_to_tfjs.ipynb**: **[New]** Exporting PyTorch models to TensorFlow.js for on-device mobile inference.

---

## 📱 Mobile Apps

This project includes two versions of the React Native app:

### 1. Standard (Online Mode)
- **Location**: `/DigitRecognizer`
- **Architecture**: Mobile App -> Flask API (`inference_server.py`) -> PyTorch Model.
- **Best for**: Learning about AI APIs and backend integration.

### 2. Local AI (Offline Mode)
- **Location**: `/DigitRecognizerLocal`
- **Architecture**: Mobile App -> Local TensorFlow.js Model.
- **Best for**: Learning about on-device AI and minimizing latency/costs.
- **Run**: `cd DigitRecognizerLocal && npx expo start`

---

## 📱 Digit Recognizer Mobile App

The mobile app connects to a Python inference server to predict digits in real-time.

### 1. Start the Inference Server
```bash
# In the root 'ai-tutorial' directory
uv run inference_server.py
```

### 2. Run the App
```bash
cd DigitRecognizer

# Install JS dependencies
npm install

# Start Expo (Scan QR code with Expo Go app)
npx expo start
```

> [!TIP]
> **Physical Device Testing**: The app now automatically detects your computer's IP address. Just ensure both the phone and computer are on the same Wi-Fi network.

---

## 🏗️ Architecture

-   **Model**: 3-layer fully-connected network (SimpleNN) trained on MNIST.
-   **Backend**: Flask API served via `uv`.
-   **Frontend**: React Native using `react-native-svg` and `react-native-view-shot`.
-   **Stylus Support**: Fully optimized for Samsung S-Pen input on the S24 Ultra.

## 🔧 Troubleshooting

-   **Inference Server**: Ensure the server is running on port 5000 before pressing "Predict" in the app.
-   **Network Error**: If using a physical device, ensure both the phone and Mac are on the same Wi-Fi network and the IP address is correctly configured.
-   **Inaccurate Predictions**: The model expects white digits on a black background. Our server handles this automatically by inverting your drawing before inference.

## 🎓 Learning Outcomes

-   Building and training neural networks with **PyTorch**.
-   Image preprocessing for Computer Vision.
-   Creating a production-ready **Flask API** for AI models.
-   Integrating AI into **React Native** mobile applications.

## 📜 License

MIT
