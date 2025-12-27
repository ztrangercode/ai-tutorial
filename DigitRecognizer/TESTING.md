# Digit Recognizer - Testing Guide

## 🚀 Quick Start

### 1. Start the Inference Server

In the `ai-tutorial` directory:

```bash
cd /Users/fredrik/source/ai-tutorial
source venv/bin/activate
python inference_server.py
```

You should see:
```
✓ Model loaded successfully!
✓ Server ready to accept requests
Starting server on http://0.0.0.0:5000
```

### 2. Update API URL (Physical Device Only)

If testing on your Samsung Galaxy S24 Ultra (not emulator):

1. Find your Mac's IP address:
   ```bash
   ipconfig getifaddr en0
   ```

2. Edit `DigitRecognizer/api/inference.ts`:
   - Change line 15 from `'http://10.0.2.2:5000'`
   - To `'http://YOUR_MAC_IP:5000'` (e.g., `'http://192.168.1.100:5000'`)

### 3. Run the React Native App

In a new terminal:

```bash
cd /Users/fredrik/source/ai-tutorial/DigitRecognizer
npm run android
```

## ✅ Testing Checklist

### Test 1: Drawing Works
- [ ] Open app on device
- [ ] Draw on canvas with S-Pen
- [ ] Verify strokes appear smoothly
- [ ] Tap "Clear" button
- [ ] Verify canvas clears

### Test 2: Prediction Works
- [ ] Draw digit "3"
- [ ] Tap "Predict" button
- [ ] Verify prediction shows "3"
- [ ] Check confidence score is high (>80%)

### Test 3: All Digits
Test each digit 0-9:
- [ ] 0
- [ ] 1
- [ ] 2
- [ ] 3
- [ ] 4
- [ ] 5
- [ ] 6
- [ ] 7
- [ ] 8
- [ ] 9

### Test 4: Error Handling
- [ ] Stop the inference server
- [ ] Try to predict
- [ ] Verify error message appears
- [ ] Restart server
- [ ] Verify predictions work again

## 🐛 Troubleshooting

### "Failed to connect to server"
- Ensure inference server is running
- Check firewall isn't blocking port 5000
- Verify API_BASE_URL in `api/inference.ts` is correct

### Drawing not appearing
- Try restarting the app
- Check console for errors

### Predictions are wrong
- Make sure digits are drawn clearly
- Try drawing larger
- Ensure model file exists at `models/mnist_model.pth`

## 📝 Next Steps

Once testing is complete:
1. Commit the working code
2. Evaluate if suitable for a lesson
3. Consider adding features:
   - On-device inference (ONNX)
   - History of predictions
   - Confidence threshold warnings
