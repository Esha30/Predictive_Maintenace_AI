"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, LayoutDashboard, Settings, ActivitySquare, AlertTriangle, PlayCircle } from "lucide-react";

export default function Sidebar() {
  const pathname = usePathname();

  const links = [
    { name: "Dashboard", href: "/", icon: LayoutDashboard },
    { name: "Machines", href: "/machines", icon: Settings },
    { name: "Analytics", href: "/analytics", icon: ActivitySquare },
    { name: "Run Prediction", href: "/predict", icon: PlayCircle },
    { name: "Model Info", href: "/model", icon: AlertTriangle },
  ];

  return (
    <div className="w-64 bg-slate-800 border-r border-slate-700 h-full flex flex-col">
      <div className="p-6 flex items-center gap-3">
        <Activity className="text-indigo-400" size={28} />
        <h1 className="text-xl font-bold tracking-tight text-white">Predictive AI</h1>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-2">
        {links.map((link) => {
          const Icon = link.icon;
          const active = pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href));
          return (
            <Link
              key={link.name}
              href={link.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                active
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400 hover:bg-slate-700 hover:text-slate-200"
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{link.name}</span>
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
        © 2026 AI Maintenance Inc.
      </div>
    </div>
  );
}
