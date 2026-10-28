"use client";

import { useState, useEffect } from "react";
import { predictAndAnalyze, fetchMachines } from "@/lib/api";
import { Machine, FullAnalysisResponse } from "@/types";
import { Activity, AlertTriangle, ShieldCheck } from "lucide-react";

export default function PredictPage() {
  const [machines, setMachines] = useState<Machine[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FullAnalysisResponse | null>(null);
  
  const [formData, setFormData] = useState({
    machine_id: "",
    machine_type: "M",
    air_temp_k: 298.0,
    process_temp_k: 308.0,
    rotational_speed_rpm: 1500.0,
    torque_nm: 40.0,
    tool_wear_min: 0.0,
  });

  useEffect(() => {
    fetchMachines().then(setMachines).catch(console.error);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await predictAndAnalyze({
        ...formData,
        machine_id: formData.machine_id === "" ? undefined : formData.machine_id
      });
      setResult(res);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500 max-w-5xl">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Run Prediction</h1>
        <p className="text-slate-400">Input manual sensor values to run inference on the XGBoost model.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <form onSubmit={handleSubmit} className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-lg space-y-5">
          <h3 className="text-xl font-bold text-white border-b border-slate-700 pb-3">Sensor Input</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-slate-300">Machine</label>
              <select name="machine_id" value={formData.machine_id} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500">
                <option value="">-- Unregistered Machine --</option>
                {machines.map(m => <option key={m.machine_id} value={m.machine_id}>{m.machine_id} - {m.machine_name}</option>)}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-slate-300">Type Quality</label>
              <select name="machine_type" value={formData.machine_type} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500">
                <option value="L">Low (L)</option>
                <option value="M">Medium (M)</option>
                <option value="H">High (H)</option>
              </select>
            </div>
            
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-slate-300">Air Temp [K]</label>
              <input type="number" step="0.1" name="air_temp_k" value={formData.air_temp_k} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500" required />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-slate-300">Process Temp [K]</label>
              <input type="number" step="0.1" name="process_temp_k" value={formData.process_temp_k} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500" required />
            </div>
            
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-slate-300">Rotational Speed [rpm]</label>
              <input type="number" step="1" name="rotational_speed_rpm" value={formData.rotational_speed_rpm} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500" required />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-slate-300">Torque [Nm]</label>
              <input type="number" step="0.1" name="torque_nm" value={formData.torque_nm} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500" required />
            </div>

            <div className="space-y-1.5 col-span-2">
              <label className="text-sm font-medium text-slate-300">Tool Wear [min]</label>
              <input type="number" step="1" name="tool_wear_min" value={formData.tool_wear_min} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500" required />
            </div>
          </div>

          <button type="submit" disabled={loading} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50 mt-4">
            {loading ? "Analyzing..." : "Analyze Sensor Data"}
          </button>
        </form>

        {result && (
          <div className="space-y-6 animate-in slide-in-from-right-4 duration-500">
            <div className={`p-6 rounded-xl border shadow-lg ${result.prediction === 'FAILURE_RISK' ? 'bg-rose-500/10 border-rose-500/20' : 'bg-emerald-500/10 border-emerald-500/20'}`}>
              <div className="flex items-center gap-3 mb-2">
                {result.prediction === 'FAILURE_RISK' ? <AlertTriangle className="text-rose-400" size={28} /> : <ShieldCheck className="text-emerald-400" size={28} />}
                <h3 className={`text-2xl font-bold ${result.prediction === 'FAILURE_RISK' ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {result.prediction === 'FAILURE_RISK' ? 'Failure Risk Detected' : 'Normal Operation'}
                </h3>
              </div>
              <div className="flex items-end gap-2 mb-4">
                <span className="text-5xl font-black text-white">{(result.failure_probability * 100).toFixed(1)}%</span>
                <span className="text-slate-300 mb-1">probability</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Risk Level</span>
                  <span className="font-bold text-white">{result.risk_level}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Anomaly Detection</span>
                  <span className={`font-bold ${result.is_anomaly ? 'text-amber-400' : 'text-emerald-400'}`}>{result.anomaly_label} (Score: {result.anomaly_score})</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-lg">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2"><Activity size={18} className="text-indigo-400" /> Recommended Action</h4>
              <p className="text-slate-300">{result.recommended_action}</p>
            </div>
            
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-lg">
              <h4 className="font-bold text-white mb-4">Top Contributing Factors (SHAP)</h4>
              <div className="space-y-3">
                {result.shap_explanation.top_features.slice(0,4).map((f, i) => (
                  <div key={i} className="flex justify-between items-center text-sm">
                    <span className="text-slate-300 font-medium">{f.feature} <span className="text-slate-500">({f.feature_value})</span></span>
                    <span className={`font-bold ${f.direction === 'positive' ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {f.direction === 'positive' ? '+' : ''}{f.shap_value.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
