"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function OutlierChart({ data }: { data: Array<{ metric: string; count: number }> }) {
  return <div className="h-72 w-full"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 10, right: 8, bottom: 30, left: 0 }}><CartesianGrid stroke="#d9d7d0" vertical={false} /><XAxis dataKey="metric" angle={-22} textAnchor="end" height={58} tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><YAxis tick={{ fill: "#686861", fontSize: 11 }} tickLine={false} axisLine={false} /><Tooltip cursor={{ fill: "#ebe9e2" }} contentStyle={{ border: "1px solid #d9d7d0", borderRadius: 0, background: "#f4f3ef", fontSize: 12 }} /><Bar dataKey="count" fill="#1746a2" maxBarSize={44} /></BarChart></ResponsiveContainer></div>;
}