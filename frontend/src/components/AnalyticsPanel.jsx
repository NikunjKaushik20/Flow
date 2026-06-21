import React from 'react';
import { BarChart3 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend,
} from 'recharts';

const COLORS = {
  critical: '#FF5D5D',
  moderate: '#FFB84D',
  minor: '#47D18C',
  focus: '#9EC1FF',
  muted: '#3A4052',
  text: '#A7B0BE',
  bg: '#111218',
};

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#1a1a2e', border: '1px solid #3A4052',
      borderRadius: 6, padding: '8px 12px', fontSize: 12,
    }}>
      <div style={{ color: '#F5F7FA', marginBottom: 4, fontWeight: 600 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || COLORS.focus }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
};

const AnalyticsPanel = ({ data }) => {
  if (!data) {
    return (
      <div className="panel" style={{ padding: 'var(--space-5)', textAlign: 'center' }}>
        <BarChart3 size={48} className="text-muted" />
        <h2 className="text-md" style={{ marginTop: 'var(--space-3)' }}>Connecting to API...</h2>
        <p className="text-muted text-sm">Start the backend server to see analytics</p>
      </div>
    );
  }

  const { hourly_distribution, daily_distribution, cause_distribution,
    duration_by_cause, severity_distribution, corridor_distribution, monthly_trend } = data;

  const severityPie = [
    { name: '<30min', value: severity_distribution?.['<30min'] || 0, color: COLORS.minor },
    { name: '30min-2hr', value: severity_distribution?.['30min-2hr'] || 0, color: COLORS.moderate },
    { name: '2hr+', value: severity_distribution?.['2hr+'] || 0, color: COLORS.critical },
  ];

  return (
    <div className="analytics-grid">
      {/* Hourly Distribution */}
      <div className="panel chart-card">
        <h3 className="label">Incident Distribution by Hour</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={hourly_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis dataKey="hour" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill={COLORS.focus} radius={[3, 3, 0, 0]} name="Incidents" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Severity Distribution */}
      <div className="panel chart-card">
        <h3 className="label">Severity Distribution</h3>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={severityPie} cx="50%" cy="50%" innerRadius={50} outerRadius={80}
              paddingAngle={3} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {severityPie.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Daily Distribution */}
      <div className="panel chart-card">
        <h3 className="label">Incidents by Day of Week</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={daily_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis dataKey="day" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill={COLORS.moderate} radius={[3, 3, 0, 0]} name="Incidents" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Event Cause Distribution */}
      <div className="panel chart-card">
        <h3 className="label">Event Cause Breakdown</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={(cause_distribution || []).slice(0, 8)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis type="number" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <YAxis type="category" dataKey="cause" tick={{ fill: COLORS.text, fontSize: 11 }} width={120} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill={COLORS.critical} radius={[0, 3, 3, 0]} name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Duration by Cause */}
      <div className="panel chart-card">
        <h3 className="label">Mean Duration by Cause (minutes)</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={(duration_by_cause || []).slice(0, 8)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis type="number" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <YAxis type="category" dataKey="cause" tick={{ fill: COLORS.text, fontSize: 11 }} width={120} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="mean" fill={COLORS.minor} radius={[0, 3, 3, 0]} name="Mean (min)" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Corridors */}
      <div className="panel chart-card">
        <h3 className="label">Top Corridors by Incident Count</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={(corridor_distribution || []).slice(0, 8)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis type="number" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <YAxis type="category" dataKey="corridor" tick={{ fill: COLORS.text, fontSize: 11 }} width={120} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill={COLORS.focus} radius={[0, 3, 3, 0]} name="Incidents" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AnalyticsPanel;
