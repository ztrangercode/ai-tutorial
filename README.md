# AI Tutorial: Handwritten Digit Recognition with MNIST

Learn how to train a neural network to recognize handwritten digits using the MNIST dataset.

## Prerequisites

- Basic Python knowledge
- Computer with macOS or Windows
- Python 3.8 or later (Python 3.14+ works with PyTorch)

## Installation Instructions

### macOS

1. **Install Python 3.8 or later**
   ```bash
   # Check if Python is installed
   python3 --version
   
   # If not installed, install via Homebrew
   brew install python3
   ```

2. **Create a virtual environment**
   ```bash
   # Navigate to the project directory
   cd ai-tutorial
   
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision numpy matplotlib jupyter scikit-learn seaborn scipy
   ```

### Windows

1. **Install Python 3.8 or later**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation:
     ```cmd
     python --version
     ```

2. **Create a virtual environment**
   ```cmd
   # Navigate to the project directory
   cd ai-tutorial
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```cmd
   pip install --upgrade pip
   pip install torch torchvision numpy matplotlib jupyter scikit-learn seaborn scipy
   ```

## Verify Installation

Run this command to verify PyTorch is installed correctly:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Getting Started

### Interactive Notebooks

1. Activate your virtual environment (if not already activated)
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Work through the exercises in order:
   - `01_hello_pytorch.ipynb` - PyTorch basics
   - `02_load_mnist.ipynb` - Load and explore data
   - `03_create_model.ipynb` - Build a neural network
   - `04_train_model.ipynb` - Train the model
   - `05_evaluate_model.ipynb` - Evaluate performance
   - `06_inference.ipynb` - Make predictions

### Advanced: Vision Transformer (Python Script)

Train a Vision Transformer model from the command line:
```bash
python train_transformer.py
```

This demonstrates:
- Patch-based image processing
- Multi-head self-attention
- Positional embeddings
- Modern transformer architecture

## What You'll Learn

- Setting up a Python ML development environment
- Loading and exploring the MNIST dataset
- Building neural networks with PyTorch (fully-connected and transformers)
- Training and evaluating models
- Making predictions on handwritten digits
- **Advanced**: Understanding attention mechanisms and Vision Transformers

## Resources

- [PyTorch Documentation](https://pytorch.org/tutorials/)
- [PyTorch MNIST Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues installing PyTorch:
- PyTorch supports Python 3.8+ including the latest versions
- For GPU support on NVIDIA cards, visit [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific installation commands
- On macOS with Apple Silicon (M1/M2/M3), PyTorch includes MPS (Metal Performance Shaders) support automatically

### Virtual Environment Not Activating

- Make sure you're in the project directory
- On Windows, you may need to enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## License

MIT
