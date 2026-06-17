'use client';

import { AlertTriangle, Calendar, CheckCircle2, MapPin, Trash2, ShieldAlert, MoreVertical } from 'lucide-react';
import { Alert } from '../../types';

interface AlertsViewProps {
    alerts: Alert[];
    onDelete?: (id: number) => void;
    onUpdateStatus?: (id: number, status: string) => void;
    onFilterChange?: (status?: string, severity?: string) => void;
}

export const AlertsView: React.FC<AlertsViewProps> = ({
    alerts,
    onDelete,
    onUpdateStatus,
    onFilterChange
}) => {
    const handleFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value;
        if (val === 'All Alerts') {
            onFilterChange?.();
        } else if (val === 'Resolved') {
            onFilterChange?.('resolved');
        } else if (val === 'Critical') {
            onFilterChange?.(undefined, 'critical');
        } else if (val === 'Active') {
            onFilterChange?.('active');
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h2 className="text-2xl font-bold text-white">Alert History</h2>
                    <p className="text-slate-400">Recent incidents and automated flags</p>
                </div>
                <div className="flex gap-2">
                    <select
                        onChange={handleFilterChange}
                        className="bg-slate-900 border border-white/10 text-slate-300 text-sm rounded-lg px-4 py-2 outline-none focus:border-orange-500"
                    >
                        <option>All Alerts</option>
                        <option>Active</option>
                        <option>Critical</option>
                        <option>Resolved</option>
                    </select>
                </div>
            </div>

            {alerts.length === 0 ? (
                <div className="text-center py-20 bg-slate-900/30 rounded-2xl border border-white/5 border-dashed">
                    <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                        <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                    </div>
                    <h3 className="text-white font-medium">All Clear</h3>
                    <p className="text-slate-500 mt-1">No alerts recorded in the system log.</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {alerts.map((alert) => (
                        <div key={alert.id} className="glass-panel glass-panel-hover rounded-xl p-5 border border-white/5 transition-all group">
                            <div className="flex items-start justify-between">
                                <div className="flex items-start gap-4">
                                    <div className={`p-3 rounded-xl ${alert.severity === 'critical' ? 'bg-red-500/20 text-red-500 animate-pulse' :
                                        alert.status === 'resolved' ? 'bg-emerald-500/10 text-emerald-400' :
                                            'bg-orange-500/10 text-orange-400'
                                        }`}>
                                        {alert.severity === 'critical' ? <ShieldAlert className="w-6 h-6" /> : <AlertTriangle className="w-6 h-6" />}
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-3 mb-1">
                                            <h3 className="font-semibold text-white">{alert.fire_type} Detected</h3>
                                            <div className="flex gap-2">
                                                <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${alert.status === 'resolved' ? 'bg-emerald-500/20 text-emerald-400' :
                                                    alert.status === 'critical' || alert.severity === 'critical' ? 'bg-red-600 text-white' :
                                                        'bg-orange-500/20 text-orange-400'
                                                    }`}>
                                                    {alert.status}
                                                </span>
                                                {alert.severity === 'critical' && (
                                                    <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-red-600 text-white">
                                                        CRITICAL
                                                    </span>
                                                )}
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-4 text-sm text-slate-400">
                                            <div className="flex items-center gap-1.5">
                                                <MapPin className="w-3.5 h-3.5" />
                                                {alert.lat && alert.lng ? (
                                                    <a
                                                        href={`https://www.google.com/maps/search/?api=1&query=${alert.lat},${alert.lng}`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-orange-400 hover:text-orange-300 hover:underline flex items-center gap-1"
                                                    >
                                                        {alert.location}
                                                    </a>
                                                ) : (
                                                    alert.location
                                                )}
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <Calendar className="w-3.5 h-3.5" />
                                                {alert.timestamp}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex flex-col items-end gap-2">
                                    <div className="text-right">
                                        <div className="text-lg font-bold text-white">
                                            {(alert.confidence * 100).toFixed(0)}%
                                        </div>
                                        <div className="text-xs text-slate-500">Confidence</div>
                                    </div>

                                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        {alert.status !== 'resolved' && (
                                            <button
                                                onClick={() => onUpdateStatus?.(alert.id, 'resolved')}
                                                className="p-1.5 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20 rounded-lg transition-colors"
                                                title="Resolve"
                                            >
                                                <CheckCircle2 className="w-4 h-4" />
                                            </button>
                                        )}
                                        <button
                                            onClick={() => onDelete?.(alert.id)}
                                            className="p-1.5 bg-red-500/10 text-red-500 hover:bg-red-500/20 rounded-lg transition-colors"
                                            title="Delete"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
