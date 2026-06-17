// Type definitions for Fire Detection System

export interface DetectionResult {
    fire_detected: boolean;
    confidence: number;
    fire_type: string | null;
    fire_type_probs: { [key: string]: number } | null;
    timestamp: string;
    bounding_boxes: number[][];
    attention_weights?: {
        yolo: number;
        vit: number;
        optical_flow: number;
    };
}

export interface Alert {
    id: number;
    fire_type: string;
    location: string;
    confidence: number;
    timestamp: string;
    iso_timestamp: string;
    status: 'active' | 'resolved' | 'critical';
    severity: 'normal' | 'warning' | 'critical';
    lat?: number;
    lng?: number;
    image_path?: string;
}

export type AppTab = 'detect' | 'monitor' | 'alerts' | 'settings';

export interface ApiStatus {
    status: 'online' | 'offline' | 'checking';
    latency: number;
}
