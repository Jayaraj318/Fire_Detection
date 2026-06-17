'use client';

import React, { useEffect, useState } from 'react';
import { ScanEye, Search, Database, Zap, ShieldAlert } from 'lucide-react';

interface ProcessingViewProps {
    image: string;
}

const STEPS = [
    { text: "Scanning thermal signature...", icon: ScanEye },
    { text: "Isolating visual anomalies...", icon: Search },
    { text: "Processing YOLOv8 detection...", icon: Database },
    { text: "Analyzing ViT attention maps...", icon: Zap },
    { text: "Calculating hazard probability...", icon: ShieldAlert },
];

export const ProcessingView: React.FC<ProcessingViewProps> = ({ image }) => {
    const [currentStep, setCurrentStep] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep((prev) => (prev < STEPS.length - 1 ? prev + 1 : prev));
        }, 800);
        return () => clearInterval(interval);
    }, []);

    const StepIcon = STEPS[currentStep].icon;

    return (
        <div className="flex flex-col items-center justify-center w-full max-w-2xl mx-auto animate-in fade-in duration-500 py-12">
            <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl border border-orange-500/20 bg-slate-900 group">
                {/* Background blurred image */}
                <img
                    src={image}
                    alt="Analyzing"
                    className="absolute inset-0 w-full h-full object-cover blur-sm opacity-40 scale-105 transition-transform duration-1000"
                />

                {/* Main clear image overlay */}
                <div className="absolute inset-0 p-8 flex items-center justify-center">
                    <img
                        src={image}
                        alt="Subject"
                        className="w-full h-full object-contain drop-shadow-2xl z-10"
                    />
                </div>

                {/* Grid Overlay */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(15,23,42,0)_50%,rgba(0,0,0,0.4)_50%),linear-gradient(90deg,rgba(255,255,255,0.03),rgba(255,255,255,0.03))] bg-[length:100%_4px,50px_100%] z-20 pointer-events-none mix-blend-overlay"></div>

                {/* Scanning Laser Effect */}
                <div className="scan-line"></div>

                {/* Tech Overlay UI */}
                <div className="absolute inset-0 z-30 p-6 flex flex-col justify-between pointer-events-none">
                    <div className="flex justify-between items-start">
                        <div className="flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded border border-orange-500/30 text-orange-400 text-xs font-mono">
                            <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></span>
                            ACTIVE_SCAN
                        </div>
                        <div className="bg-slate-950/80 backdrop-blur-md px-2 py-1 rounded border border-white/10 text-[10px] text-slate-400 font-mono">
                            SEQ: {Math.random().toString(36).substring(7).toUpperCase()}
                        </div>
                    </div>

                    {/* Bounding Box Animation */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border border-orange-500/50 rounded-lg animate-pulse opacity-50">
                        <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-orange-500 -mt-0.5 -ml-0.5"></div>
                        <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-orange-500 -mt-0.5 -mr-0.5"></div>
                        <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-orange-500 -mb-0.5 -ml-0.5"></div>
                        <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-orange-500 -mb-0.5 -mr-0.5"></div>
                    </div>

                    <div className="flex justify-between items-end">
                        <div className="flex flex-col gap-2">
                            <div className="text-[10px] text-orange-400 font-mono tracking-wider">NEURAL_NET_V8</div>
                            <div className="w-32 h-1 bg-slate-800 rounded-full overflow-hidden">
                                <div className="h-full bg-orange-500 animate-loading"></div>
                            </div>
                        </div>
                        <div className="text-[10px] text-slate-400 font-mono text-right">
                            <div>HYBRID_MODEL</div>
                            <div>CONFIDENCE_CALC...</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-8 flex flex-col items-center w-full max-w-sm">
                <div className="flex items-center gap-3 text-orange-500 mb-4 h-8">
                    <StepIcon className="w-5 h-5 animate-bounce" />
                    <h2 className="text-lg font-medium text-white tracking-wide transition-all duration-300">
                        {STEPS[currentStep].text}
                    </h2>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-orange-600 to-red-500 transition-all duration-500 ease-out shadow-[0_0_10px_rgba(249,115,22,0.5)]"
                        style={{ width: `${((currentStep + 1) / STEPS.length) * 100}%` }}
                    ></div>
                </div>

                <div className="grid grid-cols-5 gap-1 mt-2 w-full">
                    {STEPS.map((_, i) => (
                        <div
                            key={i}
                            className={`h-0.5 rounded-full transition-colors duration-300 ${i <= currentStep ? 'bg-orange-500/50' : 'bg-slate-800'
                                }`}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};
