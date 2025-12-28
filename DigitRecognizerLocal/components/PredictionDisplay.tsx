/**
 * Prediction Display Component
 * Shows the predicted digit and confidence
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { PredictionResult } from '../api/inference';

interface PredictionDisplayProps {
    prediction: PredictionResult | null;
    loading: boolean;
}

export const PredictionDisplay: React.FC<PredictionDisplayProps> = ({
    prediction,
    loading,
}) => {
    if (loading) {
        return (
            <View style={styles.container}>
                <Text style={styles.loadingText}>Analyzing...</Text>
            </View>
        );
    }

    if (!prediction) {
        return (
            <View style={styles.container}>
                <Text style={styles.placeholderText}>Draw a digit above and tap Predict!</Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <Text style={styles.label}>Prediction:</Text>
            <Text style={styles.digit}>{prediction.digit}</Text>

            <View style={styles.probabilitiesContainer}>
                <Text style={styles.probabilitiesTitle}>All Probabilities:</Text>
                {prediction.probabilities.map((prob, index) => (
                    <View key={index} style={styles.probabilityRow}>
                        <Text style={styles.probabilityDigit}>{index}:</Text>
                        <View style={styles.probabilityBarContainer}>
                            <View
                                style={[
                                    styles.probabilityBar,
                                    {
                                        width: `${prob * 100}%`,
                                        backgroundColor:
                                            index === prediction.digit ? '#4CAF50' : '#2196F3',
                                    },
                                ]}
                            />
                        </View>
                        <Text style={styles.probabilityText}>{(prob * 100).toFixed(1)}%</Text>
                    </View>
                ))}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        padding: 20,
        alignItems: 'center',
    },
    label: {
        fontSize: 18,
        color: '#666',
        marginBottom: 8,
    },
    digit: {
        fontSize: 72,
        fontWeight: 'bold',
        color: '#2196F3',
        marginBottom: 24,
    },
    loadingText: {
        fontSize: 20,
        color: '#666',
        fontStyle: 'italic',
    },
    placeholderText: {
        fontSize: 16,
        color: '#999',
        textAlign: 'center',
    },
    probabilitiesContainer: {
        width: '100%',
        marginTop: 16,
    },
    probabilitiesTitle: {
        fontSize: 16,
        fontWeight: '600',
        marginBottom: 12,
        color: '#333',
    },
    probabilityRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 8,
    },
    probabilityDigit: {
        width: 30,
        fontSize: 14,
        color: '#666',
    },
    probabilityBarContainer: {
        flex: 1,
        height: 20,
        backgroundColor: '#E0E0E0',
        borderRadius: 4,
        overflow: 'hidden',
        marginHorizontal: 8,
    },
    probabilityBar: {
        height: '100%',
        borderRadius: 4,
    },
    probabilityText: {
        width: 50,
        fontSize: 12,
        color: '#666',
        textAlign: 'right',
    },
});
