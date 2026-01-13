#!/bin/bash

# Demo Startup Script
# This script starts both the inference server (backend) and the mobile app

echo "🚀 Starting AI Tutorial Demo..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start the inference server in the background
echo "📡 Starting inference server..."
cd "$SCRIPT_DIR"
uv run python inference_server.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo ""

# Wait a bit for the server to start
sleep 3

# Start the mobile app
echo "📱 Starting mobile app..."
cd "$SCRIPT_DIR/DigitRecognizer"
npm start &
APP_PID=$!
echo "   Mobile app PID: $APP_PID"
echo ""

echo "✅ Demo environment started!"
echo ""
echo "To stop the demo:"
echo "  kill $BACKEND_PID $APP_PID"
echo ""
echo "Or press Ctrl+C and run:"
echo "  pkill -f 'inference_server.py'"
echo "  pkill -f 'expo start'"
echo ""

# Wait for user to press Ctrl+C
wait
