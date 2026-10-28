"use client";

import { useEffect, useState } from "react";
import { fetchOverviewStats, fetchRecentPredictions } from "@/lib/api";
import { OverviewStats, FullAnalysisResponse } from "@/types";
import { Activity, AlertTriangle, ShieldCheck, Cpu } from "lucide-react";
import Link from "next/link";

export default function Dashboard() {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [recent, setRecent] = useState<FullAnalysisResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchOverviewStats(), fetchRecentPredictions()])
      .then(([statsData, recentData]) => {
        setStats(statsData);
        setRecent(recentData);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="flex h-full items-center justify-center text-slate-400">Loading dashboard...</div>;
  }

  if (!stats) return <div className="text-red-400">Failed to load stats.</div>;

  const getRiskColor = (level: string) => {
    switch (level) {
      case "LOW": return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
      case "MEDIUM": return "bg-amber-500/10 text-amber-400 border-amber-500/20";
      case "HIGH": return "bg-orange-500/10 text-orange-400 border-orange-500/20";
      case "CRITICAL": return "bg-red-500/10 text-red-400 border-red-500/20";
      default: return "bg-slate-500/10 text-slate-400 border-slate-500/20";
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Fleet Overview</h1>
        <p className="text-slate-400">Monitor your industrial machine health and recent AI predictions.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">Total Machines</p>
              <h3 className="text-3xl font-bold text-white">{stats.total_machines}</h3>
            </div>
            <div className="p-3 bg-indigo-500/20 rounded-lg">
              <Cpu className="text-indigo-400" size={24} />
            </div>
          </div>
        </div>
        
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">Healthy</p>
              <h3 className="text-3xl font-bold text-emerald-400">{stats.healthy}</h3>
            </div>
            <div className="p-3 bg-emerald-500/20 rounded-lg">
              <ShieldCheck className="text-emerald-400" size={24} />
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">At Risk (Warn/Crit)</p>
              <h3 className="text-3xl font-bold text-orange-400">{stats.warning + stats.critical}</h3>
            </div>
            <div className="p-3 bg-orange-500/20 rounded-lg">
              <AlertTriangle className="text-orange-400" size={24} />
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">Active Anomalies</p>
              <h3 className="text-3xl font-bold text-rose-400">{stats.anomalies_detected}</h3>
            </div>
            <div className="p-3 bg-rose-500/20 rounded-lg">
              <Activity className="text-rose-400" size={24} />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 shadow-lg overflow-hidden">
        <div className="p-6 border-b border-slate-700 flex justify-between items-center">
          <h2 className="text-xl font-bold text-white">Recent Predictions</h2>
          <Link href="/machines" className="text-sm text-indigo-400 hover:text-indigo-300 font-medium">View All Machines →</Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-800/50 text-slate-400 text-sm border-b border-slate-700">
                <th className="px-6 py-4 font-medium">Machine ID</th>
                <th className="px-6 py-4 font-medium">Time</th>
                <th className="px-6 py-4 font-medium">Failure Prob.</th>
                <th className="px-6 py-4 font-medium">Risk Level</th>
                <th className="px-6 py-4 font-medium">Anomaly</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {recent.map((pred, i) => (
                <tr key={i} className="hover:bg-slate-700/30 transition-colors">
                  <td className="px-6 py-4 font-medium text-white">
                    {pred.machine_id ? <Link href={`/machines/${pred.machine_id}`} className="hover:underline">{pred.machine_id}</Link> : "Unknown"}
                  </td>
                  <td className="px-6 py-4 text-slate-300 text-sm">{new Date(pred.timestamp).toLocaleString()}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <div 
                          className="bg-indigo-500 h-2 rounded-full" 
                          style={{ width: `${pred.failure_probability * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-slate-300">{(pred.failure_probability * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-2.5 py-1 rounded-full text-xs font-bold border ${getRiskColor(pred.risk_level)}`}>
                      {pred.risk_level}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    {pred.is_anomaly ? (
                      <span className="inline-flex items-center gap-1 text-rose-400 text-sm font-medium">
                        <AlertTriangle size={14} /> Yes
                      </span>
                    ) : (
                      <span className="text-slate-400 text-sm">No</span>
                    )}
                  </td>
                </tr>
              ))}
              {recent.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-8 text-center text-slate-500">
                    No predictions found. Run a prediction to see results here.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
