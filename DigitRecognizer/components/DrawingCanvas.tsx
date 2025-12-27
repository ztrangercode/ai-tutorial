/**
 * Drawing Canvas Component
 * Supports touch and S-Pen input for drawing digits
 */

import React, { useRef, useState, useCallback } from 'react';
import {
    View,
    StyleSheet,
    TouchableOpacity,
    Text,
    PanResponder,
    GestureResponderEvent,
} from 'react-native';
import Svg, { Path } from 'react-native-svg';
import ViewShot from 'react-native-view-shot';

const CANVAS_SIZE = 280; // Display size

interface Point {
    x: number;
    y: number;
}

interface Stroke {
    points: Point[];
    path: string;
}

interface DrawingCanvasProps {
    onClear?: () => void;
    onDrawingChange?: (hasDrawing: boolean) => void;
    canvasRef?: React.RefObject<ViewShot | null>;
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
    onClear,
    onDrawingChange,
    canvasRef: externalRef,
}) => {
    const internalRef = useRef<ViewShot>(null);
    const canvasRef = externalRef || internalRef;

    const [strokes, setStrokes] = useState<Stroke[]>([]);
    const [currentStroke, setCurrentStroke] = useState<Point[]>([]);

    const createPathData = (points: Point[]): string => {
        if (points.length === 0) return '';

        let path = `M ${points[0].x} ${points[0].y}`;
        for (let i = 1; i < points.length; i++) {
            path += ` L ${points[i].x} ${points[i].y}`;
        }
        return path;
    };

    const handleRelease = useCallback(() => {
        setCurrentStroke(current => {
            if (current.length > 0) {
                const path = createPathData(current);
                setStrokes(prev => {
                    const newStrokes = [...prev, { points: current, path }];
                    if (prev.length === 0) {
                        onDrawingChange?.(true);
                    }
                    return newStrokes;
                });
            }
            return [];
        });
    }, [onDrawingChange]);

    const panResponder = useRef(
        PanResponder.create({
            onStartShouldSetPanResponder: () => true,
            onMoveShouldSetPanResponder: () => true,

            onPanResponderGrant: (evt: GestureResponderEvent) => {
                const { locationX, locationY } = evt.nativeEvent;
                setCurrentStroke([{ x: locationX, y: locationY }]);
            },

            onPanResponderMove: (evt: GestureResponderEvent) => {
                const { locationX, locationY } = evt.nativeEvent;
                setCurrentStroke(prev => [...prev, { x: locationX, y: locationY }]);
            },

            onPanResponderRelease: handleRelease,
            onPanResponderTerminate: handleRelease,
        }),
    ).current;

    const clearCanvas = () => {
        setStrokes([]);
        setCurrentStroke([]);
        onDrawingChange?.(false);
        onClear?.();
    };

    const hasDrawing = strokes.length > 0 || currentStroke.length > 0;

    return (
        <View style={styles.container}>
            <View style={styles.canvasContainer}>
                <ViewShot
                    ref={canvasRef}
                    options={{ format: 'png', quality: 1.0, result: 'base64' }}>
                    <View style={styles.drawingArea} {...panResponder.panHandlers}>
                        <Svg
                            width={CANVAS_SIZE}
                            height={CANVAS_SIZE}
                            style={{ backgroundColor: '#FFFFFF' }}>
                            {/* Render completed strokes */}
                            {strokes.map((stroke, index) => (
                                <Path
                                    key={`stroke-${index}`}
                                    d={stroke.path}
                                    stroke="#000000"
                                    strokeWidth={20}
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    fill="none"
                                />
                            ))}
                            {/* Render current stroke */}
                            {currentStroke.length > 0 && (
                                <Path
                                    d={createPathData(currentStroke)}
                                    stroke="#000000"
                                    strokeWidth={20}
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    fill="none"
                                />
                            )}
                        </Svg>
                    </View>
                </ViewShot>
            </View>

            <TouchableOpacity
                style={[styles.clearButton, !hasDrawing && styles.clearButtonDisabled]}
                onPress={clearCanvas}
                disabled={!hasDrawing}>
                <Text style={styles.clearButtonText}>Clear</Text>
            </TouchableOpacity>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
    },
    canvasContainer: {
        width: CANVAS_SIZE,
        height: CANVAS_SIZE,
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        borderWidth: 2,
        borderColor: '#333',
        overflow: 'hidden',
        elevation: 4,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
    },
    drawingArea: {
        width: CANVAS_SIZE,
        height: CANVAS_SIZE,
    },
    clearButton: {
        marginTop: 16,
        backgroundColor: '#FF6B6B',
        paddingHorizontal: 32,
        paddingVertical: 12,
        borderRadius: 8,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.2,
        shadowRadius: 1.41,
    },
    clearButtonDisabled: {
        backgroundColor: '#CCC',
        opacity: 0.5,
    },
    clearButtonText: {
        color: '#FFFFFF',
        fontSize: 16,
        fontWeight: '600',
    },
});
