"use client";

import { useEffect, useState } from "react";
import { fetchAnalyticsFailures, fetchAnomalyStats } from "@/lib/api";
import { AnalyticsFailures, AnomalyStats } from "@/types";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Activity, AlertTriangle, ShieldCheck } from "lucide-react";

export default function AnalyticsPage() {
  const [failures, setFailures] = useState<AnalyticsFailures | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchAnalyticsFailures(), fetchAnomalyStats()])
      .then(([f, a]) => {
        setFailures(f);
        setAnomalies(a);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400">Loading analytics...</div>;
  if (!failures || !anomalies) return <div className="text-red-400">Failed to load analytics data.</div>;

  const riskData = [
    { name: "Low", value: failures.risk_distribution["LOW"] || 0, color: "#10b981" },
    { name: "Medium", value: failures.risk_distribution["MEDIUM"] || 0, color: "#f59e0b" },
    { name: "High", value: failures.risk_distribution["HIGH"] || 0, color: "#f97316" },
    { name: "Critical", value: failures.risk_distribution["CRITICAL"] || 0, color: "#ef4444" },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Analytics</h1>
        <p className="text-slate-400">Fleet-wide failure distribution, risk analysis, and anomaly statistics.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <h3 className="text-lg font-bold text-white mb-6">Failure Risk Distribution</h3>
          <div className="h-[250px] w-full flex items-center justify-center">
            {failures.total_analyzed > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={riskData} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={5} dataKey="value" label={({name, percent}: {name?: string; percent?: number}) => `${name ?? ''} ${((percent ?? 0) * 100).toFixed(0)}%`}>
                    {riskData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-slate-500">No prediction data available.</div>
            )}
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <h3 className="text-lg font-bold text-white mb-6">Daily Failure Predictions (Last 7 Days)</h3>
          <div className="h-[250px] w-full">
            {failures.daily_failures_7d.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={failures.daily_failures_7d}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="date" stroke="#94a3b8" tickFormatter={(val) => val.split('-').slice(1).join('/')} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip cursor={{ fill: '#334155', opacity: 0.4 }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                  <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} name="Failures" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500">No recent failure predictions.</div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg col-span-2">
          <h3 className="text-lg font-bold text-white mb-4">Anomaly Detection Stats</h3>
          <p className="text-sm text-slate-400 mb-6">Unusual sensor patterns detected across the fleet (Isolation Forest).</p>
          
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1 flex items-center gap-1.5"><ShieldCheck size={16} className="text-emerald-400" /> Normal</div>
              <div className="text-2xl font-bold text-white">{anomalies.normal_count}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1 flex items-center gap-1.5"><AlertTriangle size={16} className="text-rose-400" /> Anomalies</div>
              <div className="text-2xl font-bold text-white">{anomalies.anomalies_count}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1 flex items-center gap-1.5"><Activity size={16} className="text-indigo-400" /> Avg Score</div>
              <div className="text-2xl font-bold text-white">{anomalies.avg_anomaly_score.toFixed(3)}</div>
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
          <h3 className="text-lg font-bold text-white mb-4">Top Anomalous Machines</h3>
          <ul className="space-y-3">
            {anomalies.top_anomalous_machines.map((m, i) => (
              <li key={i} className="flex justify-between items-center text-sm">
                <span className="text-slate-300 font-medium">Machine <span className="text-white font-bold">{m.machine_id}</span></span>
                <span className="px-2.5 py-1 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded-full font-bold">
                  {m.anomaly_count} alerts
                </span>
              </li>
            ))}
            {anomalies.top_anomalous_machines.length === 0 && (
              <li className="text-slate-500 text-sm">No anomalies detected recently.</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}
