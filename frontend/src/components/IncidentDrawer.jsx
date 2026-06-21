import React from 'react';
import { AlertTriangle, Info, Users, Navigation, Clock, MapPin, Shield, BarChart3 } from 'lucide-react';

const IncidentDrawer = ({ data, simulation, optimization }) => {
  const {
    severity, pred_duration_bucket, probabilities,
    impact_score, manpower, diversion, action_window, event_title,
    congestion_impact, barricade_plan, event_mode, diversion_route
  } = data;

  const getSeverityClass = (sev) => {
    switch (sev) {
      case 'CRITICAL': return 'bg-critical';
      case 'MODERATE': return 'bg-moderate';
      case 'MINOR': return 'bg-minor';
      default: return '';
    }
  };

  const getSeverityText = (sev) => {
    switch (sev) {
      case 'CRITICAL': return 'severity-critical';
      case 'MODERATE': return 'severity-moderate';
      case 'MINOR': return 'severity-minor';
      default: return '';
    }
  };

  return (
    <div className="drawer-content">
      {/* Header */}
      <div style={{ marginBottom: 'var(--space-4)' }}>
        <h2 className="text-md font-semibold" style={{ marginBottom: 'var(--space-2)' }}>{event_title}</h2>
        <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center', flexWrap: 'wrap' }}>
          <div className={`status-pill ${getSeverityClass(severity)}`}>
            {severity} SEVERITY
          </div>
          {event_mode && (
            <span className="text-xs" style={{
              padding: '2px 8px', borderRadius: 'var(--radius-sm)',
              background: event_mode === 'Planned Event' ? 'rgba(158, 193, 255, 0.2)' : 'rgba(255, 184, 77, 0.2)',
              color: event_mode === 'Planned Event' ? '#9EC1FF' : '#FFB84D',
              fontWeight: 600
            }}>
              {event_mode}
            </span>
          )}
        </div>
      </div>
      
      {/* Congestion Impact Index */}
      {congestion_impact && (
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <h3 className="label" style={{ marginBottom: 'var(--space-2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span><BarChart3 size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Congestion Impact Forecast</span>
            <span className="text-xs" style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px', fontWeight: 'normal' }}>Heuristic</span>
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-2)' }}>
            <div className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>BPR Vehicle-Delay Hours</div>
              <div className={`text-sm font-semibold ${getSeverityText(severity)}`}>
                {congestion_impact.bpr_vehicle_delay_hours?.toLocaleString()}
              </div>
            </div>
            <div className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Impact Index</div>
              <div className={`text-sm font-semibold ${getSeverityText(severity)}`}>
                {congestion_impact.congestion_impact_index}/100
              </div>
            </div>
            <div className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Affected Corridors</div>
              <div className="text-sm font-semibold">{congestion_impact.affected_corridors}</div>
            </div>
            <div className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Heuristic Queue Length</div>
              <div className="text-sm font-semibold">{congestion_impact.heuristic_queue_length_km} km</div>
            </div>
          </div>
        </div>
      )}

      {/* Prediction Probabilities */}
      <div style={{ marginBottom: 'var(--space-4)' }}>
        <h3 className="label">Prediction Probabilities</h3>
        <div className="flex-col gap-2">
          {[
            { label: 'Critical (2hr+)', value: probabilities.critical_2hr_plus, cls: 'bg-critical' },
            { label: 'Moderate (30m–2hr)', value: probabilities.moderate_30min_2hr, cls: 'bg-moderate' },
            { label: 'Minor (<30m)', value: probabilities.minor_under_30min, cls: 'bg-minor' },
          ].map(p => (
            <div key={p.label} className="prob-card">
              <div className="flex justify-between text-xs" style={{ marginBottom: 4 }}>
                <span>{p.label}</span>
                <span>{(p.value * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-bar-container">
                <div className={`prob-bar ${p.cls}`} style={{ width: `${p.value * 100}%` }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Impact Score */}
      <div style={{ marginBottom: 'var(--space-4)' }}>
        <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
          <h3 className="label" style={{ margin: 0 }}>Impact Score</h3>
          <span className={`text-md font-semibold ${getSeverityText(severity)}`}>{impact_score} / 10</span>
        </div>
        <div className="impact-meter">
          {[1,2,3].map(i => <div key={i} className={`impact-segment ${impact_score >= i ? 'active-minor' : ''}`}></div>)}
          {[4,5,6].map(i => <div key={i} className={`impact-segment ${impact_score >= i ? 'active-mod' : ''}`}></div>)}
          {[7,8,9,10].map(i => <div key={i} className={`impact-segment ${impact_score >= i ? 'active-crit' : ''}`}></div>)}
        </div>
      </div>

      {/* Simulation Results */}
      {simulation && (
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <h3 className="label" style={{ marginBottom: 'var(--space-2)' }}>
            <Navigation size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
            Network Impact
          </h3>
          <div className="rec-card">
            <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Corridors Affected</div>
            <div className="text-sm font-semibold">{simulation.total_affected} corridors</div>
          </div>
          <div className="rec-card">
            <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Vehicle-Hours of Delay</div>
            <div className="text-sm font-semibold">{simulation.total_delay_vehicle_hours?.toLocaleString()}</div>
          </div>
          {simulation.affected_corridors?.slice(0, 3).map(ac => (
            <div key={ac.corridor} className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>
                {ac.corridor} {ac.is_event_corridor ? '★' : ''}
              </div>
              <div className="text-sm">Peak delay: <b className={ac.peak_delay >= 3 ? 'severity-critical' : ac.peak_delay >= 1.5 ? 'severity-moderate' : 'severity-minor'}>{ac.peak_delay}x</b></div>
            </div>
          ))}
        </div>
      )}

      {/* Resource Allocation */}
      {optimization && (
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <h3 className="label" style={{ marginBottom: 'var(--space-2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span><Users size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Resource Allocation</span>
            <span className="text-xs" style={{ background: 'rgba(71,209,140,0.15)', color: '#47D18C', padding: '2px 6px', borderRadius: '4px', fontWeight: 'normal' }}>Data-Derived</span>
          </h3>
          <div className="rec-card">
            <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Officers Deployed</div>
            <div className="text-sm font-semibold">{optimization.total_officers_used} / {optimization.total_officers_available}</div>
          </div>
          <div className="rec-card">
            <div className="text-xs text-muted" style={{ marginBottom: 4 }}>Barricade Sets</div>
            <div className="text-sm font-semibold">{optimization.total_barricades_used} / {optimization.total_barricades_available}</div>
          </div>
          {optimization.allocations?.slice(0, 3).map(a => (
            <div key={a.corridor} className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>{a.corridor} {a.is_event_corridor ? '★' : ''}</div>
              <div className="text-xs">{a.officers_assigned} officers • Delay: {a.peak_delay_before}x → {a.peak_delay_after}x ({a.delay_reduction_pct}% ↓)</div>
            </div>
          ))}
        </div>
      )}

      {/* Barricade Plan */}
      {barricade_plan && barricade_plan.total_units > 0 && (
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <h3 className="label" style={{ marginBottom: 'var(--space-2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span><Shield size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Barricade Deployment ({barricade_plan.total_units} units)</span>
            <span className="text-xs" style={{ background: 'rgba(71,209,140,0.15)', color: '#47D18C', padding: '2px 6px', borderRadius: '4px', fontWeight: 'normal' }}>Data-Derived</span>
          </h3>
          {barricade_plan.placements?.map((p, idx) => (
            <div key={idx} className="rec-card">
              <div className="text-xs text-muted" style={{ marginBottom: 4 }}>
                <MapPin size={10} style={{ marginRight: 2, verticalAlign: 'middle' }} />
                {p.description}
              </div>
              <div className="text-xs">{p.units} units • {p.lat.toFixed(4)}°N, {p.lon.toFixed(4)}°E</div>
            </div>
          ))}
        </div>
      )}

      {/* Operational Recommendations */}
      <div style={{ marginBottom: 'var(--space-3)' }}>
        <h3 className="label" style={{ marginBottom: 'var(--space-2)' }}>Operational Directives</h3>
        <div className="rec-card">
          <div className="text-xs text-muted" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span>Manpower</span>
            <span style={{ background: 'rgba(71,209,140,0.15)', color: '#47D18C', padding: '1px 4px', borderRadius: '3px', fontSize: '10px' }}>Data-Derived</span>
          </div>
          <div className="text-sm font-medium">{manpower}</div>
        </div>
        <div className="rec-card">
          <div className="text-xs text-muted" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span>Diversion</span>
            <span style={{ background: 'rgba(71,209,140,0.15)', color: '#47D18C', padding: '1px 4px', borderRadius: '3px', fontSize: '10px' }}>Data-Derived</span>
          </div>
          <div className="text-sm font-medium">{diversion}</div>
          {diversion_route?.streets?.length > 0 && (
            <div className="text-xs text-muted" style={{ marginTop: 4 }}>
              Route: {diversion_route.streets.join(' → ')}
              {diversion_route.distance_km ? ` (${diversion_route.distance_km} km)` : ''}
            </div>
          )}
        </div>
        <div className="rec-card">
          <div className="text-xs text-muted" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span>Action Window</span>
            <span style={{ background: 'rgba(255,255,255,0.1)', padding: '1px 4px', borderRadius: '3px', fontSize: '10px' }}>Heuristic</span>
          </div>
          <div className="text-sm font-medium">{action_window}</div>
        </div>
      </div>

      <div className="explainer-block">
        <Info size={16} style={{ flexShrink: 0, marginTop: 2 }} />
        <p>
          Prediction by leakage-free ensemble. Congestion impact estimated via BPR model with {data.raw_features_used || '—'} features. Resource allocation optimized via MILP.
          {event_mode === 'Planned Event' && " (Note: Due to dataset constraints, planned event allocations rely entirely on deterministic MILP.)"}
        </p>
      </div>
    </div>
  );
};

export default IncidentDrawer;
