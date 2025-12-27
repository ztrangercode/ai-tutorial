/**
 * API Client for Digit Recognition
 */

export interface PredictionResult {
    digit: number;
    confidence: number;
    probabilities: number[];
}

export interface ApiError {
    error: string;
}

const API_BASE_URL = 'http://192.168.1.190:5000'; // Updated to your current IP
// For physical device, replace with your computer's IP: 'http://192.168.1.XXX:5000'

/**
 * Send image to inference server and get prediction
 */
import { Alert } from 'react-native';

// ...

export async function predictDigit(
    imageBase64: string,
): Promise<PredictionResult> {
    try {
        console.log(`Sending prediction request to ${API_BASE_URL}/predict`);

        // Clean the base64 string if it contains the prefix
        const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: cleanBase64,
            }),
        });

        if (!response.ok) {
            const errorData: ApiError = await response.json().catch(() => ({ error: 'Unknown server error' }));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const result: PredictionResult = await response.json();
        return result;
    } catch (error) {
        throw new Error(`Prediction failed: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * Check if the inference server is running
 */
export async function checkServerHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
        });
        return response.ok;
    } catch (error) {
        return false;
    }
}
