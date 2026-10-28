"use client";

import { useEffect, useState } from "react";
import { fetchModelInfo } from "@/lib/api";
import { ModelInfo } from "@/types";
import { Brain, CheckCircle2, AlertCircle, Check } from "lucide-react";

export default function ModelInfoPage() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelInfo()
      .then(setModel)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400">Loading model information...</div>;
  if (!model) return <div className="text-red-400">Failed to load model info. Backend may not be running.</div>;

  const metrics = model.metrics?.models?.[model.primary_model] || {};

  return (
    <div className="space-y-8 animate-in fade-in duration-500 max-w-5xl">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Model Performance</h1>
        <p className="text-slate-400">Evaluation metrics and configuration for the active ML models.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg col-span-2">
          <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-700">
            <div className="p-3 bg-indigo-500/20 rounded-lg">
              <Brain className="text-indigo-400" size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">{model.primary_model} <span className="text-sm font-normal text-slate-400 px-2 py-0.5 bg-slate-700 rounded-full ml-2">v{model.model_version}</span></h2>
              <p className="text-sm text-emerald-400 flex items-center gap-1 mt-1"><CheckCircle2 size={14} /> Active for inference</p>
            </div>
          </div>

          <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Test Set Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1">ROC-AUC</div>
              <div className="text-2xl font-bold text-white">{metrics.roc_auc?.toFixed(4) || "N/A"}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1">F1 Score</div>
              <div className="text-2xl font-bold text-white">{metrics.f1?.toFixed(4) || "N/A"}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1">PR-AUC</div>
              <div className="text-2xl font-bold text-white">{metrics.pr_auc?.toFixed(4) || "N/A"}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1">Recall</div>
              <div className="text-2xl font-bold text-white">{metrics.recall?.toFixed(4) || "N/A"}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1">Precision</div>
              <div className="text-2xl font-bold text-white">{metrics.precision?.toFixed(4) || "N/A"}</div>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="text-slate-400 text-sm mb-1">Accuracy</div>
              <div className="text-2xl font-bold text-white">{metrics.accuracy?.toFixed(4) || "N/A"}</div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Risk Thresholds</h3>
            <ul className="space-y-3">
              {Object.entries(model.risk_thresholds || {}).map(([level, thresh]) => (
                <li key={level} className="flex justify-between text-sm">
                  <span className="text-white font-medium">{level}</span>
                  <span className="text-slate-400 font-mono">{thresh}</span>
                </li>
              ))}
            </ul>
          </div>
          
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Available Models</h3>
            <ul className="space-y-2">
              {model.models_available?.map((name) => (
                <li key={name} className="flex items-center gap-2 text-sm text-slate-300">
                  <Check size={14} className="text-emerald-400" /> {name}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
      
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Feature Engineering Pipeline</h3>
        <p className="text-sm text-slate-400 mb-4">Models use {model.feature_count} features derived from the raw sensor data.</p>
        <div className="flex flex-wrap gap-2">
          {model.feature_names?.map((f) => (
            <span key={f} className="px-3 py-1 bg-slate-900 border border-slate-700 rounded-full text-xs font-mono text-indigo-300">
              {f}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
