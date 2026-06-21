import React, { useState } from 'react';
import { Users, Shield, Navigation, TrendingDown } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// Use relative URL when proxy is configured, absolute URL for direct access
const API = import.meta.env.PROD ? 'http://localhost:8000' : '';

const COLORS = {
  critical: '#FF5D5D',
  moderate: '#FFB84D',
  minor: '#47D18C',
  focus: '#9EC1FF',
  muted: '#3A4052',
  text: '#A7B0BE',
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
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
};

const ResourcePanel = ({ simulationData }) => {
  const [officers, setOfficers] = useState(20);
  const [barricades, setBarricades] = useState(10);
  const [plan, setPlan] = useState(null);
  const [whatifResult, setWhatifResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [extraOfficers, setExtraOfficers] = useState(5);

  const handleOptimize = async () => {
    if (!simulationData?.affected_corridors) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          affected_corridors: simulationData.affected_corridors,
          total_officers: officers,
          total_barricades: barricades,
        }),
      });
      if (res.ok) setPlan(await res.json());
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleWhatIf = async () => {
    if (!simulationData?.affected_corridors) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/whatif`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          affected_corridors: simulationData.affected_corridors,
          base_officers: officers,
          extra_officers: extraOfficers,
        }),
      });
      if (res.ok) setWhatifResult(await res.json());
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  if (!simulationData) {
    return (
      <div className="panel" style={{ padding: 'var(--space-5)', textAlign: 'center' }}>
        <Users size={48} className="text-muted" />
        <h2 className="text-md" style={{ marginTop: 'var(--space-3)' }}>No Active Simulation</h2>
        <p className="text-muted text-sm">Run an impact simulation first to allocate resources</p>
      </div>
    );
  }

  const getBarColor = (severity) => {
    if (severity === 'CRITICAL') return COLORS.critical;
    if (severity === 'MODERATE') return COLORS.moderate;
    return COLORS.minor;
  };

  const allocChartData = plan?.allocations?.slice(0, 10).map(a => ({
    corridor: a.corridor.length > 14 ? a.corridor.substring(0, 12) + '…' : a.corridor,
    before: a.peak_delay_before,
    after: a.peak_delay_after,
    severity: a.severity,
  })) || [];

  return (
    <div className="optimizer-grid">
      {/* Controls */}
      <div className="panel">
        <div className="section-heading">
          <Users size={18} />
          <h2>Resource Constraints</h2>
        </div>

        <div className="flex-col gap-3" style={{ marginTop: 'var(--space-3)' }}>
          <div className="form-group">
            <label className="label">Available Officers: {officers}</label>
            <input type="range" min="5" max="50" value={officers}
              onChange={e => setOfficers(parseInt(e.target.value))}
              className="range-slider" />
            <div className="flex justify-between text-xs text-muted" style={{ marginTop: 2 }}>
              <span>5</span><span>50</span>
            </div>
          </div>

          <div className="form-group">
            <label className="label">Barricade Sets: {barricades}</label>
            <input type="range" min="2" max="30" value={barricades}
              onChange={e => setBarricades(parseInt(e.target.value))}
              className="range-slider" />
            <div className="flex justify-between text-xs text-muted" style={{ marginTop: 2 }}>
              <span>2</span><span>30</span>
            </div>
          </div>

          <button className="btn-accent" onClick={handleOptimize} disabled={loading} style={{ width: '100%' }}>
            {loading ? 'Optimizing...' : 'Optimize Allocation'}
          </button>
        </div>

        {/* What-If Section */}
        <div style={{ marginTop: 'var(--space-4)', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--border-subtle)' }}>
          <div className="section-heading">
            <TrendingDown size={18} />
            <h2>What-If Analysis</h2>
          </div>
          <div className="form-group" style={{ marginTop: 'var(--space-2)' }}>
            <label className="label">Add Extra Officers: +{extraOfficers}</label>
            <input type="range" min="1" max="20" value={extraOfficers}
              onChange={e => setExtraOfficers(parseInt(e.target.value))}
              className="range-slider" />
          </div>
          <button className="btn-secondary" onClick={handleWhatIf} disabled={loading} style={{ width: '100%' }}>
            Compare Scenarios
          </button>

          {whatifResult && (
            <div className="whatif-result" style={{ marginTop: 'var(--space-3)' }}>
              <div className="rec-card">
                <div className="text-xs text-muted">Base ({officers} officers)</div>
                <div className="text-sm font-semibold">{whatifResult.marginal_improvement.base_avg_delay}x avg delay</div>
              </div>
              <div className="rec-card">
                <div className="text-xs text-muted">Enhanced (+{extraOfficers} officers)</div>
                <div className="text-sm font-semibold">{whatifResult.marginal_improvement.enhanced_avg_delay}x avg delay</div>
              </div>
              <div className="rec-card highlight">
                <div className="text-xs text-muted">Improvement</div>
                <div className="text-sm font-semibold severity-minor">{whatifResult.marginal_improvement.improvement_pct}% reduction</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Allocation Results */}
      <div className="panel">
        <div className="section-heading">
          <Shield size={18} />
          <h2>Allocation Plan</h2>
        </div>

        {plan ? (
          <>
            <div className="stat-strip compact" style={{ marginBottom: 'var(--space-3)' }}>
              <div className="stat-tile mini">
                <span>Officers</span>
                <strong>{plan.total_officers_used}/{plan.total_officers_available}</strong>
              </div>
              <div className="stat-tile mini">
                <span>Barricades</span>
                <strong>{plan.total_barricades_used}/{plan.total_barricades_available}</strong>
              </div>
              <div className="stat-tile mini">
                <span>Reserve</span>
                <strong>{plan.officers_reserve} officers</strong>
              </div>
            </div>

            {/* Before/After Chart */}
            {allocChartData.length > 0 && (
              <div style={{ marginBottom: 'var(--space-3)' }}>
                <h3 className="label" style={{ marginBottom: 'var(--space-2)' }}>Delay Before vs After</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={allocChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
                    <XAxis type="number" tick={{ fill: COLORS.text, fontSize: 10 }} />
                    <YAxis type="category" dataKey="corridor" tick={{ fill: COLORS.text, fontSize: 10 }} width={100} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="before" fill={COLORS.critical} name="Before" radius={[0, 3, 3, 0]} barSize={8} />
                    <Bar dataKey="after" fill={COLORS.minor} name="After" radius={[0, 3, 3, 0]} barSize={8} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Allocation Table */}
            <div className="corridor-table">
              <div className="corridor-table-header" style={{ gridTemplateColumns: '1fr 70px 60px 90px 80px' }}>
                <span>Corridor</span>
                <span>Officers</span>
                <span>Barr.</span>
                <span>Delay ↓</span>
                <span>Action</span>
              </div>
              {plan.allocations?.slice(0, 12).map(a => (
                <div key={a.corridor} className="corridor-table-row" style={{ gridTemplateColumns: '1fr 70px 60px 90px 80px' }}>
                  <span>
                    {a.corridor} {a.is_event_corridor ? '★' : ''}
                  </span>
                  <b>{a.officers_assigned}</b>
                  <b>{a.barricade_sets}</b>
                  <em className={a.severity === 'CRITICAL' ? 'severity-critical' : a.severity === 'MODERATE' ? 'severity-moderate' : 'severity-minor'}>
                    -{a.delay_reduction_pct}%
                  </em>
                  <span className="text-xs">{a.severity === 'CRITICAL' ? '5 min' : a.severity === 'MODERATE' ? '15 min' : 'Monitor'}</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-muted text-sm" style={{ marginTop: 'var(--space-3)' }}>
            Adjust constraints and click "Optimize Allocation" to generate a plan
          </p>
        )}
      </div>
    </div>
  );
};

export default ResourcePanel;
