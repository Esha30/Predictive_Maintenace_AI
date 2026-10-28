export const dynamic = "force-dynamic";

import { fetchMachines } from "@/lib/api";
import { Machine } from "@/types";
import Link from "next/link";
import { Settings, CheckCircle, AlertTriangle, AlertOctagon } from "lucide-react";

export default async function MachinesPage() {
  let machines: Machine[] = [];
  try {
    machines = await fetchMachines();
  } catch (e) {
    console.error(e);
  }

  const getStatusDisplay = (status: string) => {
    switch (status) {
      case "healthy": return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"><CheckCircle size={14} /> Healthy</span>;
      case "warning": return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-orange-500/10 text-orange-400 border border-orange-500/20"><AlertTriangle size={14} /> Warning</span>;
      case "critical": return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20"><AlertOctagon size={14} /> Critical</span>;
      default: return <span className="text-slate-400">{status}</span>;
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Machines</h1>
          <p className="text-slate-400">View and manage all monitored industrial equipment.</p>
        </div>
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 shadow-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-800/50 text-slate-400 text-sm border-b border-slate-700">
                <th className="px-6 py-4 font-medium">ID</th>
                <th className="px-6 py-4 font-medium">Name</th>
                <th className="px-6 py-4 font-medium">Type</th>
                <th className="px-6 py-4 font-medium">Location</th>
                <th className="px-6 py-4 font-medium">Status</th>
                <th className="px-6 py-4 font-medium">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {machines.map((m) => (
                <tr key={m.machine_id} className="hover:bg-slate-700/30 transition-colors">
                  <td className="px-6 py-4 font-bold text-indigo-400">{m.machine_id}</td>
                  <td className="px-6 py-4 text-white font-medium">{m.machine_name}</td>
                  <td className="px-6 py-4 text-slate-300 text-sm">{m.machine_type}</td>
                  <td className="px-6 py-4 text-slate-300 text-sm">{m.location || "N/A"}</td>
                  <td className="px-6 py-4">{getStatusDisplay(m.status)}</td>
                  <td className="px-6 py-4">
                    <Link href={`/machines/${m.machine_id}`} className="text-indigo-400 hover:text-indigo-300 text-sm font-medium transition-colors">
                      View Details →
                    </Link>
                  </td>
                </tr>
              ))}
              {machines.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                    No machines found.
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
