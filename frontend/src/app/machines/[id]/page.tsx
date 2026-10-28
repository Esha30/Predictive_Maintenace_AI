"use client";

import { use, useEffect, useState } from "react";
import { fetchMachine, fetchMachinePredictions } from "@/lib/api";
import { Machine, FullAnalysisResponse } from "@/types";
import { ArrowLeft, Clock, MapPin, Activity, Settings, AlertTriangle } from "lucide-react";
import Link from "next/link";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";

export default function MachineDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [machine, setMachine] = useState<Machine | null>(null);
  const [predictions, setPredictions] = useState<FullAnalysisResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchMachine(id), fetchMachinePredictions(id)])
      .then(([m, p]) => {
        setMachine(m);
        setPredictions(p.reverse()); // Chronological for charts
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="text-slate-400">Loading machine details...</div>;
  if (!machine) return <div className="text-red-400">Machine not found.</div>;

  const latestPred = predictions.length > 0 ? predictions[predictions.length - 1] : null;

  const chartData = predictions.map((p) => ({
    time: new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    prob: parseFloat((p.failure_probability * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-10">
      <div>
        <Link href="/machines" className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition-colors mb-4">
          <ArrowLeft size={16} /> Back to Machines
        </Link>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">{machine.machine_name} <span className="text-indigo-400">({machine.machine_id})</span></h1>
            <div className="flex gap-4 text-sm text-slate-400">
              <span className="flex items-center gap-1.5"><Settings size={14} /> {machine.machine_type}</span>
              <span className="flex items-center gap-1.5"><MapPin size={14} /> {machine.location}</span>
              <span className="flex items-center gap-1.5"><Clock size={14} /> {machine.operating_hours.toLocaleString()} hrs</span>
            </div>
          </div>
          {latestPred && (
            <div className={`px-4 py-2 rounded-lg border ${
              latestPred.risk_level === 'CRITICAL' ? 'bg-red-500/10 border-red-500/20 text-red-400' :
              latestPred.risk_level === 'HIGH' ? 'bg-orange-500/10 border-orange-500/20 text-orange-400' :
              latestPred.risk_level === 'MEDIUM' ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' :
              'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            }`}>
              <div className="text-xs font-medium opacity-80 mb-0.5">Current Risk</div>
              <div className="text-xl font-bold">{latestPred.risk_level}</div>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-lg">
          <h3 className="text-lg font-bold text-white mb-4">Failure Probability Trend</h3>
          <div className="h-[300px] w-full">
            {predictions.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                    itemStyle={{ color: '#818cf8' }}
                  />
                  <Line type="monotone" dataKey="prob" name="Failure Risk (%)" stroke="#818cf8" strokeWidth={3} dot={{ fill: '#818cf8', r: 4 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-slate-500">No prediction history available.</div>
            )}
          </div>
        </div>

        {latestPred && latestPred.shap_explanation && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-lg">
            <h3 className="text-lg font-bold text-white mb-4">Latest Prediction Explanation (SHAP)</h3>
            <p className="text-sm text-slate-400 mb-6">Top contributing factors pushing the risk {latestPred.failure_probability > 0.5 ? 'higher' : 'lower'}.</p>
            <div className="h-[260px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={latestPred.shap_explanation.top_features} layout="vertical" margin={{ top: 0, right: 20, left: 40, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                  <XAxis type="number" stroke="#94a3b8" />
                  <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={100} tick={{ fontSize: 12 }} />
                  <Tooltip cursor={{ fill: '#334155', opacity: 0.4 }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                  <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                    {latestPred.shap_explanation.top_features.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#ef4444' : '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
      
      {latestPred && (
        <div className="bg-indigo-900/30 border border-indigo-500/30 rounded-xl p-6 shadow-lg">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-indigo-500/20 rounded-lg shrink-0">
              <Activity className="text-indigo-400" size={24} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white mb-1">Recommended Action</h3>
              <p className="text-indigo-200">{latestPred.recommended_action}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
