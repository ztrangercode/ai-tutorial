/**
 * Digit Recognizer App
 * Identify handwritten digits using PyTorch model
 */

import React, { useRef, useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Alert,
} from 'react-native';
import ViewShot from 'react-native-view-shot';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PredictionDisplay } from './components/PredictionDisplay';
import { predictDigit, PredictionResult } from './api/inference';

const App: React.FC = () => {
  const canvasRef = useRef<ViewShot>(null);
  const [hasDrawing, setHasDrawing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleClear = () => {
    setHasDrawing(false);
    setPrediction(null);
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      setPrediction(null);

      // Capture canvas as base64 string directly
      const base64data = await canvasRef.current?.capture?.();

      if (!base64data) {
        throw new Error('Failed to capture canvas');
      }

      // ViewShot returns raw base64, predictDigit handles it
      const result = await predictDigit(base64data);
      setPrediction(result);

    } catch (error) {
      setLoading(false);
      Alert.alert('Error', `Failed to capture/predict: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Digit Recognizer</Text>
          <Text style={styles.subtitle}>Draw a digit (0-9) below</Text>
        </View>

        <DrawingCanvas
          canvasRef={canvasRef}
          onDrawingChange={setHasDrawing}
          onClear={handleClear}
        />

        <TouchableOpacity
          style={[styles.predictButton, (!hasDrawing || loading) && styles.predictButtonDisabled]}
          onPress={handlePredict}
          disabled={!hasDrawing || loading}>
          <Text style={styles.predictButtonText}>
            {loading ? 'Analyzing...' : 'Predict Digit'}
          </Text>
        </TouchableOpacity>

        <PredictionDisplay prediction={prediction} loading={loading} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scrollContent: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  predictButton: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 48,
    paddingVertical: 16,
    borderRadius: 30,
    marginTop: 32,
    marginBottom: 24,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  predictButtonDisabled: {
    backgroundColor: '#B0BEC5',
    elevation: 0,
  },
  predictButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
});

export default App;
