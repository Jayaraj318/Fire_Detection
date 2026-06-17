/**
 * API Configuration
 * Uses environment variable NEXT_PUBLIC_API_URL
 */

export const getApiUrl = (): string => {
    if (typeof window === 'undefined') {
        // Server-side
        return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    }
    // Client-side
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
};

export const getWebSocketUrl = (): string => {
    const apiUrl = getApiUrl();
    // Convert http/https to ws/wss
    return apiUrl.replace(/^http/, 'ws');
};

export const API_URL = getApiUrl();
export const WS_URL = getWebSocketUrl();
