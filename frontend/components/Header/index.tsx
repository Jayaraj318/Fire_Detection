'use client';

import React from 'react';
import { Flame, Activity, Wifi, WifiOff } from 'lucide-react';
import { ApiStatus } from '../../types';

interface HeaderProps {
    apiStatus: ApiStatus;
    checkHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({ apiStatus, checkHealth }) => {
    return (
        <header className="sticky top-0 z-50 glass-panel border-b border-white/5 px-6 py-4 mb-8">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="relative group">
                        <div className="absolute -inset-1 bg-gradient-to-r from-orange-600 to-red-600 rounded-xl blur opacity-25 group-hover:opacity-75 transition duration-1000 group-hover:duration-200"></div>
                        <div className="relative p-2.5 bg-slate-900 border border-white/10 rounded-xl">
                            <Flame className="w-6 h-6 text-orange-500" />
                        </div>
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-white tracking-tight">Ignis <span className="text-slate-500 font-light">Detection</span></h1>
                        <p className="text-xs text-orange-400 font-bold tracking-widest animate-pulse">TIERED AI SYSTEM</p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${apiStatus.status === 'online'
                        ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                        : apiStatus.status === 'checking'
                            ? 'bg-blue-500/10 border-blue-500/20 text-blue-400'
                            : 'bg-red-500/10 border-red-500/20 text-red-400'
                        }`}>
                        <div className={`w-2 h-2 rounded-full animate-pulse ${apiStatus.status === 'online' ? 'bg-emerald-400' :
                            apiStatus.status === 'checking' ? 'bg-blue-400' : 'bg-red-400'
                            }`} />
                        <span className="text-xs font-bold uppercase tracking-wider">{apiStatus.status}</span>
                        {apiStatus.status === 'online' && (
                            <span className="text-xs opacity-60 ml-1">{apiStatus.latency}ms</span>
                        )}
                    </div>

                    <button
                        onClick={checkHealth}
                        className="p-2.5 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-all active:scale-95"
                        title="Check Connectivity"
                    >
                        {apiStatus.status === 'online' ? <Wifi className="w-5 h-5" /> : <WifiOff className="w-5 h-5" />}
                    </button>

                    <button className="p-2.5 text-slate-400 hover:text-orange-400 hover:bg-orange-500/10 rounded-lg transition-all active:scale-95">
                        <Activity className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </header>
    );
};
