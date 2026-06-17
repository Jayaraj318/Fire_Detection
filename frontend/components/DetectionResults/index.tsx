'use client';

import React from 'react';
import {
    AlertTriangle,
    ShieldCheck,
    Clock,
    Crosshair,
    Maximize2,
    Share2,
    Download
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Cell
} from 'recharts';
import { DetectionResult } from '../../types';

interface DetectionResultsProps {
    result: DetectionResult;
    imageUrl: string | null;
}

export const DetectionResults: React.FC<DetectionResultsProps> = ({ result, imageUrl }) => {

    const attentionData = result.attention_weights ? [
        { name: 'YOLO', value: result.attention_weights.yolo * 100 },
        { name: 'ViT', value: result.attention_weights.vit * 100 },
        { name: 'Opt. Flow', value: result.attention_weights.optical_flow * 100 },
    ] : [];

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Status Header */}
            <div className={`relative overflow-hidden p-6 rounded-2xl border backdrop-blur-sm transition-all duration-300 ${result.fire_detected
                ? 'bg-red-950/20 border-red-500/30 shadow-[0_0_40px_-10px_rgba(239,68,68,0.2)]'
                : 'bg-emerald-950/20 border-emerald-500/30 shadow-[0_0_40px_-10px_rgba(16,185,129,0.2)]'
                }`}>
                {/* Decorative gradients */}
                <div className={`absolute top-0 right-0 w-64 h-64 bg-gradient-to-br ${result.fire_detected ? 'from-red-600/10' : 'from-emerald-600/10'
                    } to-transparent rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none`}></div>

                <div className="relative flex items-center justify-between">
                    <div className="flex items-center gap-5">
                        <div className={`p-4 rounded-xl border ${result.fire_detected
                            ? 'bg-red-500/10 border-red-500/20 text-red-500'
                            : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500'
                            }`}>
                            {result.fire_detected ? <AlertTriangle className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
                        </div>
                        <div>
                            <h3 className={`text-2xl font-bold tracking-tight ${result.fire_detected ? 'text-white' : 'text-emerald-50'
                                }`}>
                                {result.fire_detected ? 'HAZARD DETECTED' : 'AREA SECURE'}
                            </h3>
                            <div className="flex items-center gap-2 mt-1.5">
                                <span className={`flex h-2 w-2 rounded-full ${result.fire_detected ? 'bg-red-500 animate-pulse' : 'bg-emerald-500'
                                    }`}></span>
                                <p className={`text-sm font-medium ${result.fire_detected ? 'text-red-300' : 'text-emerald-300/70'
                                    }`}>
                                    {result.fire_detected
                                        ? 'Immediate intervention recommended'
                                        : 'No thermal anomalies found'}
                                </p>
                            </div>
                        </div>
                    </div>
                    <div className="text-right">
                        <div className="flex items-baseline justify-end gap-1">
                            <span className="text-4xl font-black tracking-tighter text-white">
                                {(result.confidence * 100).toFixed(1)}
                            </span>
                            <span className="text-lg text-slate-500 font-medium">%</span>
                        </div>
                        <div className="text-xs uppercase font-bold text-slate-500 tracking-wider">Confidence Score</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Visual Analysis - The Tech Overlay Image */}
                <div className="lg:col-span-2 glass-panel rounded-2xl p-1 border border-white/10 relative group">
                    <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
                        <div className="bg-slate-950/80 backdrop-blur border border-white/10 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-300 flex items-center gap-2">
                            <Crosshair className="w-3.5 h-3.5 text-orange-400" />
                            FIRE_DETECTION
                        </div>
                    </div>

                    <div className="relative rounded-xl overflow-hidden bg-slate-950 aspect-video flex items-center justify-center">
                        {imageUrl && (
                            <div className="relative w-full h-full">
                                <img
                                    src={imageUrl}
                                    alt="Analysis"
                                    className="w-full h-full object-contain"
                                />
                                {/* Tech Grid Overlay */}
                                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none"></div>

                                {/* Bounding Boxes */}
                                {result.bounding_boxes?.map((box, idx) => (
                                    <div
                                        key={idx}
                                        className="absolute border-2 border-red-500/80 shadow-[0_0_15px_rgba(239,68,68,0.4)] group-hover:border-red-400 transition-colors"
                                        style={{
                                            left: `${box[0] * 100}%`,
                                            top: `${box[1] * 100}%`,
                                            width: `${box[2] * 100}%`,
                                            height: `${box[3] * 100}%`
                                        }}
                                    >
                                        <div className="absolute -top-7 left-0 bg-red-600/90 backdrop-blur text-white text-[10px] font-mono font-bold px-2 py-1 rounded-t flex items-center gap-2">
                                            <span>FIRE_HAZARD</span>
                                            <span className="opacity-75">{(result.confidence * 100).toFixed(0)}%</span>
                                        </div>
                                        {/* Corner markers */}
                                        <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-white"></div>
                                        <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-white"></div>
                                        <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-white"></div>
                                        <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-white"></div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Footer Info on Image */}
                        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent flex items-end justify-between">
                            <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
                                <div className="flex items-center gap-1.5">
                                    <Clock className="w-3.5 h-3.5" />
                                    <span>{new Date(result.timestamp).toLocaleTimeString()}</span>
                                </div>
                                <div className="px-2 py-0.5 rounded border border-white/10 bg-white/5">
                                    RES: 1920x1080
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Metrics Sidebar */}
                <div className="space-y-6">
                    {/* Fire Type Card */}
                    {result.fire_type && (
                        <div className="glass-panel rounded-2xl p-6 border border-white/5 relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10">
                                <AlertTriangle className="w-24 h-24 text-white" />
                            </div>
                            <h4 className="text-xs uppercase font-bold text-slate-500 tracking-wider mb-2 flex items-center gap-2">
                                <span className="w-1 h-4 bg-orange-500 rounded-full"></span>
                                Classification
                            </h4>
                            <div className="text-2xl font-bold text-white break-words relative z-10">
                                {result.fire_type}
                            </div>
                            <p className="text-sm text-slate-400 mt-1">High-confidence match detected in vector database.</p>
                        </div>
                    )}

                    {/* Model Info Card */}
                    <div className="glass-panel rounded-2xl p-6 border border-white/5 h-auto flex flex-col">
                        <h4 className="text-xs uppercase font-bold text-slate-500 tracking-wider mb-4 flex items-center justify-between">
                            Active Model
                        </h4>
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-orange-600/20 rounded-lg flex items-center justify-center text-orange-500">
                                <Maximize2 className="w-6 h-6" />
                            </div>
                            <div>
                                <div className="text-white font-bold text-lg">YOLO v8/v10</div>
                                <div className="text-slate-500 text-sm">Advanced Fire Processing</div>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <button className="flex items-center justify-center gap-2 p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 text-sm font-medium text-slate-300 transition-colors">
                            <Share2 className="w-4 h-4" /> Share
                        </button>
                        <button className="flex items-center justify-center gap-2 p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 text-sm font-medium text-slate-300 transition-colors">
                            <Download className="w-4 h-4" /> Report
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
