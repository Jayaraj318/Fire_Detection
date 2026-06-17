// Mock data for demo mode
import { DetectionResult, Alert } from '../types';

export const MOCK_DETECTION: DetectionResult = {
    fire_detected: true,
    confidence: 0.94,
    fire_type: 'Class A (Solid Combustibles)',
    fire_type_probs: {
        'Class A (Combustibles)': 0.94,
        'Class B (Flammable Liquids)': 0.04,
        'Class C (Electrical)': 0.02,
    },
    timestamp: new Date().toISOString(),
    bounding_boxes: [[0.15, 0.2, 0.35, 0.45]],
    attention_weights: {
        yolo: 0.45,
        vit: 0.35,
        optical_flow: 0.20,
    },
};

export const MOCK_ALERTS: Alert[] = [
    {
        id: 1,
        fire_type: 'Class A Fire',
        location: 'Warehouse North - Sector 4',
        confidence: 0.92,
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
        iso_timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
        status: 'active',
        severity: 'critical'
    },
    {
        id: 2,
        fire_type: 'Class B Fire',
        location: 'Loading Dock - Bay 2',
        confidence: 0.87,
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
        iso_timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
        status: 'resolved',
        severity: 'warning'
    },
    {
        id: 3,
        fire_type: 'Smoke Detection',
        location: 'Server Room - Rack 12',
        confidence: 0.78,
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
        iso_timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
        status: 'resolved',
        severity: 'normal'
    },
];
