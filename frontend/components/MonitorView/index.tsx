'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, Radio, Settings2, VideoOff, Play, Square, AlertTriangle, ShieldCheck, X, Link, Usb, Wifi, Loader2 } from 'lucide-react';
import { getWebSocketUrl } from '../../lib/apiConfig';

interface DetectionResult {
    fire_detected: boolean;
    confidence: number;
    fire_type?: string | null;
    timestamp: string;
}

type CameraSource = 'laptop' | 'usb' | 'ip' | 'rtsp';

interface CameraConfig {
    type: CameraSource;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    requiresUrl: boolean;
    placeholder?: string;
}

const CAMERA_CONFIGS: CameraConfig[] = [
    { type: 'laptop', label: 'Laptop Camera', icon: Camera, requiresUrl: false },
    { type: 'usb', label: 'External USB', icon: Usb, requiresUrl: false },
    { type: 'ip', label: 'IP Camera', icon: Wifi, requiresUrl: true, placeholder: 'http://192.168.1.100:8080/video' },
    { type: 'rtsp', label: 'RTSP Stream', icon: Link, requiresUrl: true, placeholder: 'rtsp://username:password@192.168.1.100:554/stream' }
];

interface MonitorViewProps {
    deviceLocation?: string;
    coords?: { lat: number; lng: number } | null;
}

export const MonitorView: React.FC<MonitorViewProps> = ({ deviceLocation, coords: propsCoords }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const lastFrameRef = useRef<ImageData | null>(null);

    const [isStreaming, setIsStreaming] = useState(false);
    const [hasPermission, setHasPermission] = useState<boolean | null>(null);
    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [lastDetection, setLastDetection] = useState<DetectionResult | null>(null);
    const [isVerifying, setIsVerifying] = useState(false);
    const [logs, setLogs] = useState<{ time: string; message: string; type: 'info' | 'warning' | 'success' | 'error' }[]>([]);

    // Camera source state
    const [selectedCamera, setSelectedCamera] = useState<CameraSource>('laptop');
    const [showUrlDialog, setShowUrlDialog] = useState(false);
    const [cameraUrl, setCameraUrl] = useState('');
    const [usbDeviceIndex, setUsbDeviceIndex] = useState(0);
    const [useServerStream, setUseServerStream] = useState(false);
    // Refs to avoid stale closures in capture loop
    const [coords, setCoords] = useState<{ lat: number | null; lng: number | null }>({ lat: null, lng: null });
    const isStreamingRef = useRef(isStreaming);
    const useServerStreamRef = useRef(useServerStream);
    const coordsRef = useRef(coords);

    // Sync propsCoords to local ref and state
    useEffect(() => {
        if (propsCoords) {
            setCoords(propsCoords);
            coordsRef.current = propsCoords;
        }
    }, [propsCoords]);

    useEffect(() => {
        isStreamingRef.current = isStreaming;
    }, [isStreaming]);

    useEffect(() => {
        useServerStreamRef.current = useServerStream;
    }, [useServerStream]);

    // Simplified: Coordination is now handled by parent
    useEffect(() => {
        coordsRef.current = coords;
    }, [coords]);

    const addLog = useCallback((message: string, type: 'info' | 'warning' | 'success' | 'error' = 'info') => {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev.slice(-20), { time, message, type }]);
    }, []);

    // Start client-side camera (webcam via browser)
    const startClientCamera = async () => {
        try {
            addLog('Requesting camera access...', 'info');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;

                // Ensure video plays
                videoRef.current.onloadedmetadata = () => {
                    console.log('✅ Video metadata loaded:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight);
                    videoRef.current?.play().catch(err => {
                        console.error('❌ Play error:', err);
                        addLog('Failed to play camera', 'error');
                    });
                };

                // Handle play errors
                videoRef.current.onerror = (err) => {
                    console.error('❌ Video error:', err);
                    addLog('Camera error occurred', 'error');
                };

                setHasPermission(true);
                setUseServerStream(false);
                setIsStreaming(true);
                addLog('Camera connected successfully', 'success');

                // Connect to WebSocket for detection
                connectClientWebSocket();
            }
        } catch (err) {
            console.error('Camera access denied:', err);
            setHasPermission(false);
            addLog('Camera access denied', 'error');
        }
    };

    // Start server-side camera stream (USB, IP, RTSP)
    const startServerStream = async (source: string) => {
        try {
            addLog(`Connecting to ${selectedCamera} stream...`, 'info');

            // Build WebSocket URL with source parameter for IP/RTSP cameras
            const wsBaseUrl = getWebSocketUrl();
            let wsUrl = `${wsBaseUrl}/ws/stream/${selectedCamera}`;
            if ((selectedCamera === 'ip' || selectedCamera === 'rtsp') && source) {
                wsUrl += `?url=${encodeURIComponent(source)}`;
            } else if (selectedCamera === 'usb' && source) {
                wsUrl += `?device=${encodeURIComponent(source)}`;
            }

            // Append location info to URL
            const { lat, lng } = coordsRef.current;
            if (lat !== null) wsUrl += `${wsUrl.includes('?') ? '&' : '?'}lat=${lat}`;
            if (lng !== null) wsUrl += `&lng=${lng}`;
            if (deviceLocation) wsUrl += `&location_name=${encodeURIComponent(deviceLocation)}`;

            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                addLog('WebSocket connected, starting stream...', 'info');
                // Send source info as backup (in case URL params don't work)
                ws.send(JSON.stringify({ action: 'start', source, type: selectedCamera }));
                wsRef.current = ws;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('📨 Received frame:', data.type, data.frame_id || 'status');

                    if (data.type === 'status') {
                        if (data.status === 'connected') {
                            // CRITICAL FIX: Update states with proper timing
                            setUseServerStream(true);
                            setIsStreaming(true);
                            addLog(data.message, 'success');

                        } else if (data.status === 'stopped') {
                            setIsStreaming(false);
                            setUseServerStream(false);
                            addLog('Stream stopped', 'info');
                        }
                    } else if (data.type === 'frame') {
                        // Ensure img ref is ready
                        if (!imgRef.current) {
                            console.error('❌ Img ref not ready!');
                            return;
                        }

                        if (!data.data || typeof data.data !== 'string') {
                            console.error('❌ No valid frame data! Received:', data.data?.substring?.(0, 50));
                            return;
                        }

                        // Validate base64 data
                        if (!data.data.startsWith('data:image')) {
                            console.error('❌ Invalid image data format');
                            return;
                        }

                        // Set image source - this will load the image asynchronously
                        imgRef.current.src = data.data;

                        // Update detection result
                        if (data.detection) {
                            const detection = {
                                fire_detected: data.detection.fire_detected,
                                confidence: data.detection.confidence || 0,
                                fire_type: data.detection.fire_type,
                                timestamp: data.timestamp
                            };
                            setLastDetection(detection);

                            if (data.detection.fire_detected) {
                                addLog(`🔥 Fast Detection: ${(data.detection.confidence * 100).toFixed(1)}%`, 'warning');
                            }

                            if (data.detection.yolo_suspect) {
                                setIsVerifying(true);
                            }
                        }
                    } else if (data.type === 'verification') {
                        setIsVerifying(false);
                        const detection = {
                            fire_detected: data.fire_detected,
                            confidence: data.confidence || 0,
                            fire_type: data.fire_type,
                            timestamp: data.timestamp
                        };
                        setLastDetection(detection);
                        if (data.fire_detected) {
                            addLog(`✅ AI Verified: ${data.fire_type} (${(data.confidence * 100).toFixed(1)}%)`, 'success');
                        } else {
                            addLog('ℹ️ AI confirmed area is clear.', 'info');
                        }
                    } else if (data.type === 'error' || data.error) {
                        addLog(`Error: ${data.error || data.message}`, 'error');
                        setIsStreaming(false);
                        setUseServerStream(false);
                    }
                } catch (err) {
                    console.error('❌ Failed to parse message:', err);
                    addLog('Frame parsing error', 'error');
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                addLog('WebSocket connection error', 'error');
            };

            ws.onclose = () => {
                addLog('Stream disconnected', 'info');
                wsRef.current = null;
                setIsStreaming(false);
                setUseServerStream(false);
            };
        } catch (err) {
            console.error('Failed to start stream:', err);
            addLog(`Failed to start stream: ${err}`, 'error');
        }
    };

    // Connect WebSocket for client-side camera detection
    const connectClientWebSocket = () => {
        try {
            const wsUrl = getWebSocketUrl();
            const ws = new WebSocket(`${wsUrl}/ws/video`);

            ws.onopen = () => {
                addLog('Detection WebSocket connected', 'success');
                wsRef.current = ws;
                startFrameCapture();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'detection') {
                        const detection = {
                            fire_detected: data.fire_detected || false,
                            confidence: data.confidence || 0,
                            fire_type: data.fire_type,
                            timestamp: data.timestamp
                        };
                        setLastDetection(detection);
                        setIsVerifying(false); // Pure YOLO mode — no AI verification

                        if (data.fire_detected) {
                            addLog(`🔥 Detected: ${(data.confidence * 100).toFixed(1)}%`, 'warning');
                        }
                    } else if (data.type === 'verification') {
                        setIsVerifying(false);
                        const detection = {
                            fire_detected: data.fire_detected,
                            confidence: data.confidence || 0,
                            fire_type: data.fire_type,
                            timestamp: data.timestamp
                        };
                        setLastDetection(detection);
                    }

                    // Always update canvas overlay every frame
                    if (overlayCanvasRef.current && videoRef.current) {
                        const overlayCtx = overlayCanvasRef.current.getContext('2d');
                        if (overlayCtx && videoRef.current.videoWidth > 0) {
                            const cW = videoRef.current.videoWidth;
                            const cH = videoRef.current.videoHeight;
                            overlayCanvasRef.current.width = cW;
                            overlayCanvasRef.current.height = cH;
                            overlayCtx.clearRect(0, 0, cW, cH);

                            if (data.fire_detected && data.detections && data.detections.length > 0) {
                                // 1. Draw the FROZEN frame that was analyzed
                                if (lastFrameRef.current) {
                                    overlayCtx.putImageData(lastFrameRef.current, 0, 0);
                                }

                                // 2. Draw detections on the frozen frame
                                data.detections.forEach((det: any) => {
                                    const [x1, y1, x2, y2] = det.xyxy || [];
                                    const conf = det.confidence || 0;
                                    if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) return;

                                    // Mirror for front-facing webcam (already mirrored in lastFrameRef actually if captured mirrored)
                                    // Actually, startFrameCapture captures mirrored, so mX calculation might be redundant or needed depending on capture logic.
                                    // In startFrameCapture: ctx.scale(-1, 1); ctx.drawImage(video, -canvas.width, 0); 
                                    // So the frame is ALREADY mirrored. xyxy from backend should be on mirrored frame.

                                    overlayCtx.strokeStyle = '#FF4500';
                                    overlayCtx.lineWidth = 3;
                                    overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                                    const label = `🔥 FIRE ${(conf * 100).toFixed(0)}%`;
                                    overlayCtx.font = 'bold 13px Arial';
                                    const textW = overlayCtx.measureText(label).width;
                                    overlayCtx.fillStyle = 'rgba(200, 30, 0, 0.9)';
                                    overlayCtx.fillRect(x1, y1 - 22, textW + 10, 22);
                                    overlayCtx.fillStyle = '#FFFFFF';
                                    overlayCtx.fillText(label, x1 + 5, y1 - 5);
                                });

                                // 3. Auto-clear high-visibility frozen frame after 400ms to return to live
                                setTimeout(() => {
                                    if (overlayCanvasRef.current) {
                                        const ctx = overlayCanvasRef.current.getContext('2d');
                                        if (ctx) ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
                                    }
                                }, 400);
                            }
                        }
                    }
                } catch (err) {
                    console.error('❌ Failed to parse detection:', err);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                addLog('WebSocket error', 'error');
            };

            ws.onclose = () => {
                addLog('WebSocket disconnected', 'info');
                wsRef.current = null;
            };
        } catch (err) {
            console.error('Failed to connect WebSocket:', err);
            addLog('Failed to connect WebSocket', 'error');
        }
    };

    const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Capture frames from client camera and send to server
    const startFrameCapture = () => {
        if (!isStreamingRef.current) {
            console.log('⚠️ startFrameCapture called but streaming is false');
            return;
        }

        const canvas = canvasRef.current;
        const video = videoRef.current;

        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clean up any existing interval
        if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current);
        }

        let localFrameCount = 0;
        let lastFpsUpdate = Date.now();

        const captureFrame = async () => {
            // console.debug('🔄 captureFrame tick');
            if (!isStreamingRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || useServerStreamRef.current) {
                if (captureIntervalRef.current) {
                    console.log('🛑 Stopping capture loop: isStreaming=', isStreamingRef.current, ' ws=', !!wsRef.current, ' state=', wsRef.current?.readyState, ' useServerStream=', useServerStreamRef.current);
                    clearInterval(captureIntervalRef.current);
                    captureIntervalRef.current = null;
                }
                return;
            }

            try {
                // Use coords from state (non-blocking)
                const { lat, lng } = coords;

                // Set canvas dimensions to match video
                if (video.videoWidth > 0 && video.videoHeight > 0) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;

                    // Draw the video frame to the hidden canvas
                    ctx.save();
                    ctx.scale(-1, 1); // Mirror to match video display
                    ctx.drawImage(video, -canvas.width, 0);
                    ctx.restore();

                    // Store current frame for sync later
                    lastFrameRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);

                    // Convert to base64
                    const imageData = canvas.toDataURL('image/jpeg', 0.75);
                    // console.debug('📸 Frame captured, length:', imageData.length);
                    const startTime = Date.now();

                    // Send frame data as JSON to include location
                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                        try {
                            const payload = {
                                image: imageData,
                                lat: lat || null,
                                lng: lng || null,
                                location_name: deviceLocation || null
                            };
                            wsRef.current.send(JSON.stringify(payload));
                            setLatency(Date.now() - startTime);

                            localFrameCount++;
                            if (localFrameCount % 10 === 0) {
                                console.log(`🚀 [Monitor] Frame ${localFrameCount} sent. Latency: ${Date.now() - startTime}ms`);
                            }
                        } catch (sendErr) {
                            console.error('❌ Failed to send frame:', sendErr);
                            addLog('Connection lost - stopping capture', 'error');
                            setIsStreaming(false);
                            if (captureIntervalRef.current) {
                                clearInterval(captureIntervalRef.current);
                                captureIntervalRef.current = null;
                            }
                            return;
                        }
                    } else {
                        console.warn('⚠️ WebSocket not ready. State:', wsRef.current?.readyState);
                        setIsStreaming(false);
                        if (captureIntervalRef.current) {
                            clearInterval(captureIntervalRef.current);
                            captureIntervalRef.current = null;
                        }
                        return;
                    }
                } else {
                    if (localFrameCount % 30 === 0) console.log('⏳ Waiting for video dimensions...');
                }
            } catch (err) {
                console.error('Frame capture error:', err);
            }

            // FPS calculation
            const now = Date.now();
            if (now - lastFpsUpdate >= 1000) {
                setFps(localFrameCount);
                localFrameCount = 0;
                lastFpsUpdate = now;
            }
        };

        // Start the capture loop - send frames every 200ms (5 FPS)
        console.log('🎬 Starting frame capture loop...');
        captureIntervalRef.current = setInterval(captureFrame, 200);
    };

    // Handle camera selection
    const handleCameraSelect = (type: CameraSource) => {
        if (isStreaming) {
            stopCamera();
        }

        setSelectedCamera(type);

        const config = CAMERA_CONFIGS.find(c => c.type === type);
        if (config?.requiresUrl) {
            // Reset URL input and show dialog for IP/RTSP cameras
            setCameraUrl('');
            setShowUrlDialog(true);
        } else if (type === 'laptop') {
            startClientCamera();
        } else if (type === 'usb') {
            startServerStream(String(usbDeviceIndex));
        }
    };

    // Handle URL submit for IP/RTSP cameras
    const handleUrlSubmit = () => {
        const trimmedUrl = cameraUrl.trim();
        if (!trimmedUrl) {
            addLog('Please enter a valid URL', 'error');
            return;
        }

        // Validate URL format
        if (selectedCamera === 'ip' && !trimmedUrl.startsWith('http')) {
            addLog('IP camera URL should start with http:// or https://', 'error');
            return;
        }
        if (selectedCamera === 'rtsp' && !trimmedUrl.startsWith('rtsp')) {
            addLog('RTSP stream URL should start with rtsp://', 'error');
            return;
        }

        setShowUrlDialog(false);
        startServerStream(trimmedUrl);
    };

    // Stop camera/stream
    const stopCamera = () => {
        // Stop client-side camera
        if (videoRef.current?.srcObject) {
            const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }

        // Stop server-side stream
        if (wsRef.current) {
            if (useServerStream && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ action: 'stop' }));
            }

            // Properly close WebSocket
            try {
                wsRef.current.close(1000, 'User stopped stream');
            } catch (err) {
                console.warn('Error closing WebSocket:', err);
            }
            wsRef.current = null;
        }

        // Explicitly stop the capture loop
        if (captureIntervalRef.current) {
            console.log('🛑 Manually stopping capture loop in stopCamera');
            clearInterval(captureIntervalRef.current);
            captureIntervalRef.current = null;
        }

        setIsStreaming(false);
        setUseServerStream(false);
        setLastDetection(null);
        setIsVerifying(false);
        addLog('Camera stopped', 'info');
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCamera();
        };
    }, []);

    // Start frame capture when streaming client camera
    useEffect(() => {
        if (isStreaming && !useServerStream && wsRef.current?.readyState === WebSocket.OPEN) {
            // Give video a moment to load before starting capture
            const timer = setTimeout(() => {
                if (isStreamingRef.current) {
                    startFrameCapture();
                }
            }, 300);
            return () => clearTimeout(timer);
        }
    }, [isStreaming, useServerStream]);

    // Debug state changes
    useEffect(() => {
        console.log('🔍 State Update:', {
            isStreaming,
            useServerStream,
            videoVisible: !useServerStream && isStreaming,
            imgVisible: useServerStream && isStreaming
        });
    }, [isStreaming, useServerStream]);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-12rem)] min-h-[500px]">
            {/* URL Dialog Modal */}
            {showUrlDialog && (
                <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center">
                    <div className="bg-slate-900 border border-white/10 rounded-2xl p-6 w-full max-w-md mx-4">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-bold text-white">
                                {selectedCamera === 'ip' ? 'IP Camera URL' : 'RTSP Stream URL'}
                            </h3>
                            <button onClick={() => setShowUrlDialog(false)} className="text-slate-400 hover:text-white">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <p className="text-sm text-slate-400 mb-4">
                            {selectedCamera === 'ip'
                                ? 'Enter the HTTP URL of your IP camera stream'
                                : 'Enter the RTSP URL of your IP camera or NVR'
                            }
                        </p>

                        <input
                            type="text"
                            value={cameraUrl}
                            onChange={(e) => setCameraUrl(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleUrlSubmit()}
                            placeholder={CAMERA_CONFIGS.find(c => c.type === selectedCamera)?.placeholder}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-1 focus:ring-orange-500 focus:border-orange-500 outline-none font-mono text-sm"
                        />

                        <div className="flex gap-3 mt-6">
                            <button
                                onClick={() => setShowUrlDialog(false)}
                                className="flex-1 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleUrlSubmit}
                                disabled={!cameraUrl.trim()}
                                className="flex-1 px-4 py-2 bg-orange-600 hover:bg-orange-500 rounded-lg text-white font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Connect
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Main Feed */}
            <div className="lg:col-span-2 glass-panel rounded-2xl overflow-hidden border border-white/5 relative flex flex-col">
                {/* Header Badges */}
                <div className="absolute top-4 left-4 z-10 flex gap-2">
                    {isStreaming && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 backdrop-blur-md border border-white/10">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-xs font-bold text-white uppercase tracking-wider">Live</span>
                        </div>
                    )}

                    {isVerifying && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-900/80 backdrop-blur-md border border-blue-400/30">
                            <Loader2 className="w-3 h-3 text-blue-300 animate-spin" />
                            <span className="text-xs font-bold text-blue-100 uppercase tracking-wider">Verifying...</span>
                        </div>
                    )}

                    {lastDetection?.fire_detected && (
                        <div className="bg-black/50 backdrop-blur text-white text-xs font-mono px-2 py-1 rounded uppercase">
                            {selectedCamera === 'laptop' ? 'LAPTOP_CAM' :
                                selectedCamera === 'usb' ? `USB_${usbDeviceIndex}` :
                                    selectedCamera === 'ip' ? 'IP_CAM' : 'RTSP'}
                        </div>
                    )}
                </div>

                {/* Detection Status Badge */}
                {lastDetection && (
                    <div className={`absolute top-4 right-4 z-10 px-3 py-2 rounded-lg flex items-center gap-2 ${lastDetection.fire_detected
                        ? 'bg-red-600/90 text-white animate-pulse'
                        : 'bg-emerald-600/90 text-white'
                        }`}>
                        {lastDetection.fire_detected ? (
                            <>
                                <AlertTriangle className="w-4 h-4" />
                                <span className="font-bold">FIRE: {(lastDetection.confidence * 100).toFixed(0)}%</span>
                            </>
                        ) : (
                            <>
                                <ShieldCheck className="w-4 h-4" />
                                <span className="font-medium">Area Clear</span>
                            </>
                        )}
                    </div>
                )}

                {/* Video Container */}
                <div className="flex-1 bg-slate-950 relative flex items-center justify-center group">
                    {/* Hidden canvas for frame capture - we'll draw video + detections on it */}
                    <canvas
                        ref={canvasRef}
                        style={{
                            display: 'none'
                        }}
                    />

                    {/* Client-side video (laptop/browser webcam) - Visible for display */}
                    <video
                        key="client-video"
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        style={{
                            display: (!useServerStream && isStreaming) ? 'block' : 'none',
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            transform: 'scaleX(-1)' // Mirror the video like a real camera
                        }}
                    />

                    {/* Detection overlay canvas - drawn on top of video */}
                    <canvas
                        ref={overlayCanvasRef}
                        style={{
                            display: (!useServerStream && isStreaming) ? 'block' : 'none',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            cursor: 'crosshair'
                        }}
                    />

                    {/* Server-side stream image - FIXED with inline style */}
                    <img
                        key="server-stream"
                        ref={imgRef}
                        alt="Stream"
                        onError={(e) => {
                            console.error('❌ Image failed to load:', e);
                            addLog('Failed to load frame image', 'error');
                        }}
                        onLoad={() => {
                            console.log('✅ Frame loaded successfully');
                        }}
                        style={{
                            display: (useServerStream && isStreaming) ? 'block' : 'none',
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain'
                        }}
                    />

                    {/* Overlay Grid when streaming */}
                    {isStreaming && (
                        <>
                            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none"></div>
                            <div className="absolute top-0 left-0 w-full h-1 bg-orange-500/30 shadow-[0_0_15px_rgba(249,115,22,0.5)] animate-scan pointer-events-none"></div>
                        </>
                    )}

                    {/* Placeholder when not streaming */}
                    {!isStreaming && (
                        <div className="relative z-10 text-center">
                            <VideoOff className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                            <p className="text-slate-400 mb-4">Select a camera source to begin</p>
                            <button
                                onClick={() => handleCameraSelect('laptop')}
                                className="px-6 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white font-bold rounded-xl hover:opacity-90 transition-opacity flex items-center gap-2 mx-auto"
                            >
                                <Play className="w-5 h-5" />
                                Start Laptop Camera
                            </button>
                            {hasPermission === false && (
                                <p className="text-red-400 text-sm mt-4">
                                    Camera access denied. Please allow camera permissions.
                                </p>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer Controls */}
                <div className="p-4 bg-slate-900/80 border-t border-white/5 flex items-center justify-between">
                    <div className="flex gap-4 text-xs font-mono text-slate-400">
                        <span>FPS: {isStreaming ? fps : '--'}</span>
                        <span>LATENCY: {isStreaming ? `${latency}ms` : '--'}</span>
                        <span>MODE: {useServerStream ? 'SERVER' : 'CLIENT'}</span>
                    </div>
                    <div className="flex gap-2">
                        {isStreaming ? (
                            <button
                                onClick={stopCamera}
                                className="p-2 bg-red-600/20 hover:bg-red-600/30 rounded-lg text-red-400 transition-colors flex items-center gap-2"
                            >
                                <Square className="w-4 h-4" />
                                <span className="text-xs font-medium">Stop</span>
                            </button>
                        ) : (
                            <button
                                onClick={() => handleCameraSelect(selectedCamera)}
                                className="p-2 bg-emerald-600/20 hover:bg-emerald-600/30 rounded-lg text-emerald-400 transition-colors flex items-center gap-2"
                            >
                                <Play className="w-4 h-4" />
                                <span className="text-xs font-medium">Start</span>
                            </button>
                        )}
                        <button className="p-2 hover:bg-white/10 rounded-lg text-slate-400 transition-colors">
                            <Settings2 className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
                {/* Camera Selection */}
                <div className="glass-panel rounded-2xl p-6 border border-white/5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Camera className="w-5 h-5 text-orange-400" />
                        Camera Source
                    </h3>
                    <div className="space-y-2">
                        {CAMERA_CONFIGS.map((config, i) => {
                            const Icon = config.icon;
                            const isActive = selectedCamera === config.type && isStreaming;
                            const isSelected = selectedCamera === config.type;

                            return (
                                <button
                                    key={i}
                                    onClick={() => handleCameraSelect(config.type)}
                                    disabled={isStreaming && isSelected}
                                    className={`w-full p-3 rounded-xl flex items-center justify-between transition-all ${isSelected
                                        ? 'bg-orange-500/10 border border-orange-500/30 text-white'
                                        : 'bg-slate-800/50 border border-transparent hover:bg-slate-800 text-slate-400 hover:text-slate-200'
                                        } ${isStreaming && isSelected ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                                >
                                    <div className="flex items-center gap-3">
                                        <Icon className="w-4 h-4" />
                                        <span className="text-sm font-medium">{config.label}</span>
                                    </div>
                                    {isActive && <Radio className="w-4 h-4 text-orange-500 animate-pulse" />}
                                </button>
                            );
                        })}
                    </div>

                    {/* USB Device Index */}
                    {selectedCamera === 'usb' && !isStreaming && (
                        <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                            <label className="text-xs text-slate-400 block mb-2">Device Index</label>
                            <select
                                value={usbDeviceIndex}
                                onChange={(e) => setUsbDeviceIndex(Number(e.target.value))}
                                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-white"
                            >
                                {[0, 1, 2, 3, 4].map(i => (
                                    <option key={i} value={i}>Camera {i}</option>
                                ))}
                            </select>
                        </div>
                    )}
                </div>

                {/* Detection Status */}
                {lastDetection && (
                    <div className={`glass-panel rounded-2xl p-6 border ${lastDetection.fire_detected
                        ? 'border-red-500/30 bg-red-950/20'
                        : 'border-emerald-500/30 bg-emerald-950/20'
                        }`}>
                        <h3 className="text-lg font-semibold text-white mb-4">Detection Status</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <span className="text-slate-400">Status</span>
                                <span className={lastDetection.fire_detected ? 'text-red-400 font-bold' : 'text-emerald-400'}>
                                    {lastDetection.fire_detected ? '🔥 FIRE DETECTED' : '✅ Clear'}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Confidence</span>
                                <span className="text-white font-mono">{(lastDetection.confidence * 100).toFixed(1)}%</span>
                            </div>
                            {lastDetection.fire_type && (
                                <div className="flex justify-between">
                                    <span className="text-slate-400">Type</span>
                                    <span className="text-white">{lastDetection.fire_type}</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* System Log */}
                <div className="glass-panel rounded-2xl p-6 border border-white/5">
                    <h3 className="text-lg font-semibold text-white mb-4">System Log</h3>
                    <div className="font-mono text-xs space-y-2 h-48 overflow-y-auto pr-2 custom-scrollbar">
                        {logs.length === 0 ? (
                            <div className="text-slate-500">No logs yet...</div>
                        ) : (
                            logs.map((log, i) => (
                                <div key={i} className={`flex gap-2 ${log.type === 'success' ? 'text-emerald-400' :
                                    log.type === 'warning' ? 'text-orange-400' :
                                        log.type === 'error' ? 'text-red-400' :
                                            'text-slate-400'
                                    }`}>
                                    <span className="opacity-50">[{log.time}]</span>
                                    <span>{log.message}</span>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
