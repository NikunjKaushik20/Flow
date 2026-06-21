import React, { useState } from 'react';

const ScenarioForm = ({ onSubmit, loading, corridors = [] }) => {
  const [formData, setFormData] = useState({
    event_mode: 'unplanned',
    priority: 'High',
    requires_road_closure: 'False',
    corridor: 'Hosur Road',
    event_cause: 'accident',
    description: '',
  });

  const corridorOptions = corridors.length > 0
    ? corridors.filter(c => c.name !== 'Non-corridor').map(c => c.name)
    : ['Hosur Road', 'Tumkur Road', 'Mysore Road', 'Bellary Road 1', 'ORR East 1'];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const corridor = corridors.find(c => c.name === formData.corridor);
    const payload = {
      ...formData,
      latitude: corridor?.lat || 12.9716,
      longitude: corridor?.lon || corridor?.lng || 77.5946,
      start_datetime: new Date().toISOString(),
      priority_weight: formData.priority === 'High' ? 3 : formData.priority === 'Medium' ? 2 : 1,
    };
    onSubmit(payload);
  };

  const isPlanned = formData.event_mode === 'planned';

  return (
    <form onSubmit={handleSubmit} className="flex-col gap-3">
      <div className="form-group">
        <label className="label">Event Mode</label>
        <select name="event_mode" value={formData.event_mode} onChange={handleChange} className="input-field form-select">
          <option value="unplanned">Unplanned (Reactive)</option>
          <option value="planned">Planned (Proactive)</option>
        </select>
      </div>

      <div className="form-group">
        <label className="label">Event Cause</label>
        <select name="event_cause" value={formData.event_cause} onChange={handleChange} className="input-field form-select">
          {isPlanned ? (
            <>
              <option value="procession">Procession</option>
              <option value="construction">Construction</option>
              <option value="public_event">Public Event</option>
              <option value="protest">Protest</option>
              <option value="vip_movement">VIP Movement</option>
            </>
          ) : (
            <>
              <option value="accident">Accident</option>
              <option value="vehicle_breakdown">Vehicle Breakdown</option>
              <option value="water_logging">Water Logging</option>
              <option value="tree_fall">Tree Fall</option>
              <option value="congestion">Congestion</option>
              <option value="pot_holes">Pot Holes</option>
            </>
          )}
        </select>
      </div>

      <div className="form-group">
        <label className="label">Corridor</label>
        <select name="corridor" value={formData.corridor} onChange={handleChange} className="input-field form-select">
          {corridorOptions.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label className="label">Priority</label>
        <select name="priority" value={formData.priority} onChange={handleChange} className="input-field form-select">
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>

      <div className="form-group">
        <label className="label">Road Closure Required</label>
        <select name="requires_road_closure" value={formData.requires_road_closure} onChange={handleChange} className="input-field form-select">
          <option value="False">No</option>
          <option value="True">Yes</option>
        </select>
      </div>

      <div className="form-group">
        <label className="label">Description</label>
        <textarea
          name="description"
          value={formData.description}
          onChange={handleChange}
          className="input-field"
          rows={3}
          placeholder="Describe the incident (e.g., multi-vehicle collision blocking left lane, heavy vehicle overturned with fuel spill...)"
        />
      </div>

      <div className="form-group">
        <button type="submit" className="btn-accent" style={{ width: '100%' }} disabled={loading}>
          {loading ? 'Analyzing impact...' : 'Simulate Impact'}
        </button>
      </div>
    </form>
  );
};

export default ScenarioForm;
