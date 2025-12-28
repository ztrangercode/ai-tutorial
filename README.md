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

### 3. AI Exercise Path
Work through the notebooks in the `exercises/` directory:
- `01_hello_pytorch.ipynb` - PyTorch basics
- `02_load_mnist.ipynb` - Load and explore data
- `03_create_model.ipynb` - Build a neural network
- `04_train_model.ipynb` - Train the model
- `05_evaluate_model.ipynb` - Evaluate performance
- `06_inference.ipynb` - Make predictions

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
> **Physical Device Testing**: For physical devices, you must update the `API_BASE_URL` in `DigitRecognizer/api/inference.ts` to your Mac's IP address (e.g., `http://192.168.1.100:5000`).

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
