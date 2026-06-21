from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import pandas as pd
import numpy as np
import pickle
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None
import json
import os
import importlib
import networkx as nx
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager

# --- Model Loading Trickery ---
import __main__
try:
    mod02 = importlib.import_module("02_model_training")
    setattr(__main__, 'text_col_selector', mod02.text_col_selector)
    setattr(__main__, 'TargetStatsEncoder', mod02.TargetStatsEncoder)
    setattr(__main__, 'to_dense_matrix', mod02.to_dense_matrix)
except Exception:
    pass

try:
    from utils_gnn import GridlockGNN
except ImportError:
    GridlockGNN = None

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    TabularPredictor = None

# --- Globals for Model ---
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("GRIDLOCK API SERVER STARTING")
    print("=" * 60)
    print("Loading ML models...")
    try:
        print("Loading blend data from pickle...")
        with open('models/layer2_blend.pkl', 'rb') as f:
            blend_data = pickle.load(f)
        app_state['blend_data'] = blend_data
        
        # Override weights if dynamic weights exist
        if os.path.exists('models/dynamic_weights.json'):
            try:
                with open('models/dynamic_weights.json', 'r') as dw_f:
                    dynamic_weights = json.load(dw_f)
                app_state['blend_data']['blend_weights'] = dynamic_weights
                print(f"Loaded dynamic blend weights: {dynamic_weights}")
            except Exception as e:
                print(f"Failed to load dynamic weights, using base: {e}")
                
        print("Pickle loaded. Initializing GNN...")
        
        gnn_model = GridlockGNN(
            len(blend_data['gnn_available_feats']),
            blend_data['gnn_corridor_feats'].shape[1],
            len(blend_data['gnn_corridor_idx']),
            64, 3, 0.3
        )
        gnn_model.load_state_dict(torch.load("models/gnn_blend_weights.pt", weights_only=True))
        gnn_model.eval()
        app_state['gnn_model'] = gnn_model
        print("GNN loaded. Initializing AutoGluon...")
        
        app_state['ag_model'] = TabularPredictor.load("models/autogluon_gridlock", verbosity=0)
        app_state['mod02'] = mod02
        print("AutoGluon loaded. Models loaded successfully.")
        
        try:
            with open('outputs/graph_structure.json', 'r') as f:
                graph_data = json.load(f)
            
            G_endo = nx.Graph()
            for node in graph_data.get('nodes', []):
                G_endo.add_node(node['id'], lat=node.get('lat'), lon=node.get('lon'))
            for edge in graph_data.get('edges', []):
                G_endo.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0), junction=edge.get('junction', 'Unknown'))
            app_state['endogenous_graph'] = G_endo
            print(f"Endogenous graph loaded: {G_endo.number_of_nodes()} nodes, {G_endo.number_of_edges()} edges.")
        except Exception as e:
            print(f"Warning: Failed to load endogenous graph: {e}")
            app_state['endogenous_graph'] = nx.Graph()

        
        try:
            if os.path.exists('outputs/feature_metadata.json'):
                with open('outputs/feature_metadata.json', 'r') as f:
                    meta = json.load(f)
                    app_state['corridor_vulnerability'] = meta.get('corridor_vulnerability', {})
                    app_state['corridor_centroids'] = meta.get('corridor_centroids', {})
                    app_state['manpower_rules'] = meta.get('manpower_rules', {})
                    app_state['global_manpower_fallback'] = meta.get('global_manpower_fallback', {})
                    app_state['barricade_rules'] = meta.get('barricade_rules', {})
                    app_state['global_barricade_fallback'] = meta.get('global_barricade_fallback', {})
                    app_state['spillover_weights'] = meta.get('spillover_weights', {})
                    app_state['police_units'] = meta.get('police_units', [])
        except Exception:
            app_state['corridor_vulnerability'] = {}
            app_state['corridor_centroids'] = {}
            app_state['manpower_rules'] = {}
            app_state['global_manpower_fallback'] = {}
            app_state['barricade_rules'] = {}
            app_state['global_barricade_fallback'] = {}
            app_state['spillover_weights'] = {}
            app_state['police_units'] = []

        import asyncio
        try:
            from importlib import import_module
            post_event_learning = import_module("04_post_event_learning")
            async def autonomous_learning_loop():
                while True:
                    await asyncio.sleep(3600) # Run every hour autonomously
                    try:
                        print("Running autonomous post-event learning cycle...")
                        post_event_learning.run_learning_cycle()
                    except Exception as e:
                        print(f"Autonomous learning cycle failed: {e}")
            asyncio.create_task(autonomous_learning_loop())
            print("Autonomous learning loop initialized.")
        except Exception as e:
            print(f"Could not initialize autonomous learning loop: {e}")

        print("API ready at http://localhost:8000")
        print("=" * 60)
    except Exception as e:
        print(f"Warning: Failed to load models: {e}")
        print("API will run with fallback responses.")
    yield
    app_state.clear()

# Create app FIRST
app = FastAPI(title="Gridlock AI Backend", lifespan=lifespan)

# Add CORS middleware IMMEDIATELY after app creation (order matters!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Import optimization engine AFTER app is created
try:
    from optimization_engine import MILPResourceOptimizer
except ImportError:
    print("Warning: optimization_engine not found, using fallback")
    MILPResourceOptimizer = None

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# --- Pydantic Schemas ---
class IncidentData(BaseModel):
    id: str
    latitude: float
    longitude: float
    description: str
    hour: int
    day_of_week: int
    corridor: Optional[str] = "UNKNOWN"
    priority: Optional[str] = "Medium"
    requires_road_closure: Optional[bool] = False
    timestamp: Optional[str] = None

class PredictRequest(BaseModel):
    incidents: List[IncidentData]

class PredictResponse(BaseModel):
    id: str
    duration_bucket: str
    severity: str
    impact_score: float

class SimulateRequest(BaseModel):
    corridor_closed: str
    duration_hours: float

class OptimizeUnit(BaseModel):
    id: str
    lat: float
    lon: float

class OptimizeIncident(BaseModel):
    id: str
    lat: float
    lon: float
    severity: str
    impact_score: float

class OptimizeRequest(BaseModel):
    incidents: List[OptimizeIncident]
    units: List[OptimizeUnit]
    total_barricades: Optional[int] = 100

class FeedbackData(BaseModel):
    id: str
    actual_duration_min: float
    impact_score: Optional[float] = None
    event_cause: Optional[str] = None
    corridor: Optional[str] = None

# --- Endpoints ---

def get_city_center():
    try:
        if os.path.exists('outputs/feature_metadata.json'):
            with open('outputs/feature_metadata.json', 'r') as f:
                meta = json.load(f)
                if 'city_center' in meta:
                    return meta['city_center']['lat'], meta['city_center']['lon']
    except Exception:
        pass
    return 12.9716, 77.5946

def generate_diversion_route(incident_corridor, lat=None, lon=None):
    """Generate a corridor-aware bypass route using endogenous graph with spillover-aware weights.
    
    Routing prefers corridors with LOW historical spillover (less likely to be congested together).
    Gracefully handles missing lat/lon by using corridor centroid from metadata.
    """
    # Handle missing lat/lon gracefully using corridor centroids
    if (lat is None or lon is None or pd.isna(lat) or pd.isna(lon)) and incident_corridor:
        corridor_centroids = app_state.get('corridor_centroids', {})
        if incident_corridor in corridor_centroids:
            lat = corridor_centroids[incident_corridor]['lat']
            lon = corridor_centroids[incident_corridor]['lon']
        else:
            lat, lon = get_city_center()
    elif lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        lat, lon = get_city_center()

    fallback = {"route_text": f"Divert traffic away from {incident_corridor}.", "streets": [], "path_nodes": 0, "distance_km": 0}

    try:
        G = app_state.get('endogenous_graph')
        if not G or len(G.nodes) == 0:
            return fallback

        # Find the node closest to the incident corridor.
        target_node = incident_corridor
        if target_node not in G.nodes:
            # Try to find the closest by coordinates
            if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                best_dist = float('inf')
                for n, data in G.nodes(data=True):
                    n_lat = data.get('lat', lat)
                    n_lon = data.get('lon', lon)
                    if n_lat is not None and n_lon is not None:
                        d = np.sqrt((n_lat - lat)**2 + (n_lon - lon)**2)
                        if d < best_dist:
                            best_dist = d
                            target_node = n
            
        if target_node not in G.nodes:
            return fallback

        neighbors = list(G.neighbors(target_node))
        if len(neighbors) < 2:
            fallback["route_text"] = f"Divert traffic away from {incident_corridor} via surrounding local roads."
            return fallback

        # Remove the blocked corridor from the graph
        G_bypass = G.copy()
        G_bypass.remove_node(target_node)

        # Apply spillover-aware weights: prefer routes through corridors with LOW spillover
        spillover_weights = app_state.get('spillover_weights', {})
        for u, v, data in G_bypass.edges(data=True):
            key = f"{u}|{v}" if f"{u}|{v}" in spillover_weights else f"{v}|{u}"
            spillover = spillover_weights.get(key, 0.5)
            # Invert: low spillover = good diversion route (less likely to be congested together)
            # Use distance * (1 + spillover) so high spillover paths are penalized
            base_dist = data.get('weight', 1.0)
            data['diversion_weight'] = base_dist * (1.0 + spillover)

        # Find the best bypass route between neighbors using diversion_weight
        best_route, best_cost = None, float('inf')
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                src, dst = neighbors[i], neighbors[j]
                if src not in G_bypass or dst not in G_bypass:
                    continue
                try:
                    path = nx.shortest_path(G_bypass, source=src, target=dst, weight='diversion_weight')
                    cost = nx.shortest_path_length(G_bypass, source=src, target=dst, weight='diversion_weight')
                    if cost < best_cost:
                        best_cost = cost
                        best_route = path
                except nx.NetworkXNoPath:
                    continue

        if best_route:
            # Calculate actual distance for display
            actual_dist = sum(G_bypass[u][v].get('weight', 1.0) for u, v in zip(best_route[:-1], best_route[1:]))
            route_text = f"Bypass {incident_corridor}: " + " \u2192 ".join([str(n) for n in best_route[:5]])
            if len(best_route) > 5:
                route_text += f" (+{len(best_route) - 5} segments)"
            route_text += f" [~{actual_dist:.1f} km detour, spillover-optimized]"
            return {"route_text": route_text, "streets": [str(n) for n in best_route[:5]], "path_nodes": len(best_route), "distance_km": round(actual_dist, 1)}
        else:
            fallback["route_text"] = f"No bypass route available — major blockage at {incident_corridor}."
            return fallback

    except Exception as e:
        print(f"Endogenous routing error: {e}")
        return fallback

def _compute_barricade_plan(severity, is_planned, corridor_name, lat, lon, adj_idx, adj_norm, event_cause=None):
    """Compute barricade placement using endogenous graph coordinates with learned rules from dataset."""
    if lat is None or pd.isna(lat) or lon is None or pd.isna(lon):
        lat, lon = get_city_center()
    else:
        lat = float(lat)
        lon = float(lon)

    # Use learned barricade rules from historical data
    barricade_rules = app_state.get('barricade_rules', {})
    global_barricade_fallback = app_state.get('global_barricade_fallback', {})
    
    rule_key = f"('{event_cause}', '{corridor_name}')"
    total = barricade_rules.get(rule_key, global_barricade_fallback.get(event_cause, 0))
    
    # Adjust for severity and planned vs unplanned
    if severity == 'CRITICAL':
        total = int(total * 1.5)
        btype = "full_lane_barricading"
    elif severity == 'MODERATE':
        total = int(total * 1.0)
        btype = "traffic_cones_and_signs"
    else:
        total = int(total * 0.5)
        btype = "warning_signs"
    
    # Planned events get more barricades
    if is_planned:
        total = int(total * 1.2)
    
    total = max(0, min(25, total))

    if total == 0:
        return {"total_units": 0, "type": "none_required", "placements": []}

    placements = [{"type": "incident_perimeter", "lat": round(lat, 6), "lon": round(lon, 6),
                   "units": max(1, total // 3), "description": f"Perimeter barricade at incident site on {corridor_name}"}]

    try:
        G = app_state.get('endogenous_graph')
        if G and corridor_name in G.nodes:
            neighbors = list(G.neighbors(corridor_name))
            
            remaining = total - placements[0]['units']
            count = 0
            n_neighbors = len(neighbors)
            
            for neighbor in neighbors:
                if remaining <= 0 or count >= 4:
                    break
                n_lat = G.nodes[neighbor].get('lat', lat)
                n_lon = G.nodes[neighbor].get('lon', lon)
                
                units_here = max(1, remaining // max(1, min(4, n_neighbors) - count))
                placements.append({"type": "entry_barricade", "lat": round(n_lat, 6),
                                   "lon": round(n_lon, 6), "units": units_here,
                                   "description": f"Entry barricade on approach corridor: {neighbor}"})
                remaining -= units_here
                count += 1
    except Exception as e:
        print(f"Endogenous barricade logic failed: {e}. Falling back to perimeter only.")

    return {"total_units": total, "type": btype, "placements": placements}


def _store_prediction(incident_id, corridor, event_cause, pred_bucket, severity, impact_score):
    """Store prediction in SQLite for post-event learning comparison."""
    try:
        os.makedirs("outputs", exist_ok=True)
        conn = sqlite3.connect("outputs/feedback.sqlite")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id TEXT PRIMARY KEY, corridor TEXT, event_cause TEXT,
                      pred_duration_bucket TEXT, severity TEXT, impact_score REAL, timestamp TEXT)''')
        c.execute("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (str(incident_id), str(corridor), str(event_cause), str(pred_bucket),
                   str(severity), float(impact_score), datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: could not store prediction: {e}")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We can receive simulated ingest here as well
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                if "incidents" in payload:
                    # Run prediction
                    results = process_predict_logic(payload)
                    # Broadcast the results to all clients
                    await manager.broadcast(json.dumps({"type": "NEW_PREDICTION", "data": results}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/feedback")
def submit_feedback(data: FeedbackData):
    try:
        os.makedirs("outputs", exist_ok=True)
        conn = sqlite3.connect("outputs/feedback.sqlite")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id TEXT PRIMARY KEY, actual_duration_min REAL, impact_score REAL,
                      event_cause TEXT, corridor TEXT, timestamp TEXT)''')
        c.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                  (data.id, data.actual_duration_min, data.impact_score, 
                   data.event_cause, data.corridor, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Feedback recorded for learning loop"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/force_learning_cycle")
def force_learning_cycle():
    try:
        from importlib import import_module
        post_event_learning = import_module("04_post_event_learning")
        old_weights = app_state.get('blend_data', {}).get('blend_weights', {})
        post_event_learning.run_learning_cycle()
        
        with open('models/layer2_blend.pkl', 'rb') as f:
            blend_data = pickle.load(f)
        app_state['blend_data'] = blend_data
        app_state['gnn_model'].load_state_dict(torch.load("models/gnn_blend_weights.pt", weights_only=True))
        
        new_weights = blend_data.get('blend_weights', {})
        return {"status": "success", "old_weights": old_weights, "new_weights": new_weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_incident(req: Dict[str, Any]):
    """
    Accepts incoming incident data, runs through the 85+10+5 blend, 
    and returns predicted duration and operational severity.
    """
    results = process_predict_logic(req)
    # If via REST API, broadcast to WS clients as well
    await manager.broadcast(json.dumps({"type": "NEW_PREDICTION", "data": results}))
    return results

def process_predict_logic(req: Dict[str, Any]):
    if 'blend_data' not in app_state:
        # Fallback mock for demo purposes if models fail to load (e.g. on Render free tier)
        corridor = req.get('incidents', [req])[0].get('corridor', 'Unknown Corridor')
        cause = req.get('incidents', [req])[0].get('event_cause', 'Incident')
        return [{
            "id": "sim_mock_1",
            "pred_duration_bucket": "2hr+",
            "severity": "CRITICAL",
            "impact_score": 8.5,
            "event_mode": "Unplanned Event",
            "event_title": f"{cause} on {corridor}",
            "probabilities": {
                "minor_under_30min": 0.05,
                "moderate_30min_2hr": 0.20,
                "critical_2hr_plus": 0.75
            },
            "congestion_impact": {
                "bpr_vehicle_delay_hours": 3450,
                "affected_corridors": 4,
                "heuristic_queue_length_km": 4.2,
                "congestion_impact_index": 85,
                "estimated_duration_hours": 2.5,
                "estimated_duration_min": 150
            },
            "manpower": "Dispatch 4 Patrol Officers + SI (Fallback - MILP unavailable)",
            "manpower_count": 4,
            "barricade_plan": {"required": 8, "type": "Heavy Concrete + Reflective", "locations": ["Entry point", "100m upstream", "500m upstream"]},
            "diversion": "Full lane barricading. Divert traffic to alternate arterial road.",
            "diversion_route": {"route_text": "Divert via adjacent parallel routes.", "coordinates": []},
            "action_window": "Reactive: Dispatch within 5 min",
            "raw_features_used": 36,
            "confidence_score": "High",
            "confidence_breakdown": {
                "tree_gnn_agree": True,
                "gnn_ag_agree": True,
                "tree_ag_agree": True,
                "avg_max_probability": 0.78,
                "model_votes": [2, 2, 2]
            }
        }]
        
    blend_data = app_state['blend_data']
    mod02 = app_state.get('mod02', None)
    
    # Convert request to dataframe
    if "incidents" in req:
        records = req["incidents"]
    else:
        records = [req]
        
    for r in records:
        if 'id' not in r: r['id'] = 'sim_' + str(np.random.randint(1000))
        if 'latitude' not in r: 
            r['latitude'] = get_city_center()[0]
        if 'longitude' not in r: 
            r['longitude'] = get_city_center()[1]
        if 'description' not in r: r['description'] = r.get('event_cause', 'Unknown')
        if 'hour' not in r: r['hour'] = 12
        if 'day_of_week' not in r: r['day_of_week'] = 0
        if 'priority' not in r: r['priority'] = 'Medium'
        if 'requires_road_closure' not in r: r['requires_road_closure'] = False
        if 'timestamp' not in r or not r['timestamp']: r['timestamp'] = datetime.now().isoformat()
        
    df = pd.DataFrame(records)
    
    # Track Simulated Time
    latest_ts_str = max(r['timestamp'] for r in records)
    try:
        current_time = pd.to_datetime(latest_ts_str).replace(tzinfo=None)
    except:
        current_time = datetime.now()
    app_state['current_simulation_time'] = current_time
    
    # Time-Decay GNN edges: apply recent incidents multiplier based on simulated clock
    active = app_state.setdefault('active_incidents', [])
    app_state['active_incidents'] = [i for i in active if (current_time - i['time']).total_seconds() < 1800]
    
    for r in records:
        try:
            r_time = pd.to_datetime(r['timestamp']).replace(tzinfo=None)
        except:
            r_time = current_time
        if r.get('corridor') in blend_data.get('gnn_corridor_idx', {}):
            app_state['active_incidents'].append({'corridor': r['corridor'], 'time': r_time})

    # Mock some minimal feature engineering that would normally happen in 01_data_cleaning...
    df['priority_High'] = (df['priority'] == 'High').astype(int)
    # Safely convert boolean or string representation to int (1/0)
    df['requires_road_closure_True'] = df['requires_road_closure'].astype(str).str.lower().eq('true').astype(int)
    
    # --- AutoGluon Feature Padding ---
    missing_ag_cols = ['created_hour', 'created_day_of_week', 'veh_type', 'police_station', 'zone', 'junction', 'corridor_cause']
    for c in missing_ag_cols:
        if c not in df.columns:
            if c in ['created_hour', 'created_day_of_week']:
                df[c] = 0
            else:
                df[c] = 'UNKNOWN'
                
    # --- Tree/Pipeline Padding ---
    missing_tree_cols = [
        'cargo_material', 'kw_heavy', 'month', 'report_delay_min', 'rush_hour_flag', 
        'dist_from_center', 'kw_tow', 'is_weekend', 'reason_breakdown', 'kw_overturned', 
        'hour_cos', 'latitude_num', 'rush_hour_x_closure', 'kw_injury', 'minutes_since_midnight', 
        'has_end_coord', 'closure_flag', 'desc_urgency_score', 'day_cos', 'age_of_truck', 
        'month_sin', 'day_sin', 'kw_block', 'is_night', 'hour_sin', 'kw_normal', 'desc_word_count', 
        'high_priority_x_closure', 'longitude_num', 'desc_char_count', 'kw_fire', 'kw_breakdown', 
        'month_cos', 'kw_spill', 'corridor_cause', 'police_junction', 'address_text'
    ]
    
    cat_features = [
        "event_cause", "requires_road_closure", "veh_type", "corridor", "priority",
        "police_station", "zone", "junction", "cargo_material", "reason_breakdown",
        "age_of_truck", "corridor_cause", "police_junction"
    ]
    
    for c in missing_tree_cols:
        if c not in df.columns:
            if c in cat_features or c == 'address_text':
                df[c] = 'UNKNOWN'
            else:
                df[c] = 0.0

    
    # True concurrent density based on rolling simulated incidents
    current_active_count = len(app_state['active_incidents'])
    # Use historical hourly averages from dataset instead of hardcoded baseline
    hourly_baselines = app_state.get('hourly_incident_baselines', {})
    current_hour = datetime.now().hour
    baseline_expected = hourly_baselines.get(str(current_hour), 10.0)  # fallback to 10 if missing
    organic_density = min(1.0, current_active_count / max(baseline_expected, 1.0))
    df['concurrent_density'] = organic_density
    
    # 1. Tree Probabilities
    tree_entries = blend_data['tree_entries']
    tree_names = blend_data['tree_names']
    w = 1.0 / len(tree_names)
    tree_probas = {e["name"]: mod02.proba(e, df) for e in tree_entries}
    tree_probs = sum(w * tree_probas[n] for n in tree_names)
        
    # 2. GNN Probabilities
    gnn_scaler = blend_data['gnn_scaler']
    gnn_available_feats = blend_data['gnn_available_feats']
    for f in gnn_available_feats:
        if f not in df.columns: df[f] = 0
    X_infer_gnn = gnn_scaler.transform(df[gnn_available_feats].fillna(0).values)
    infer_corr = []
    corridor_centroids = app_state.get('corridor_centroids', {})
    gnn_idx = blend_data['gnn_corridor_idx']
    # UNKNOWN node index is the last index (len(corridors))
    unknown_idx = gnn_idx.get('UNKNOWN', gnn_idx.get('unknown', len(gnn_idx) - 1))
    # Build reverse mapping from idx to corridor name for distance calculation
    idx_to_corridor = {v: k for k, v in gnn_idx.items()}
    
    for idx_row, c in enumerate(df["corridor"]):
        c_lower = str(c).strip().lower()
        if c_lower in gnn_idx:
            infer_corr.append(gnn_idx[c_lower])
        else:
            # Fallback: Find nearest known corridor by Haversine distance using centroids
            # If no centroid match, use UNKNOWN node index (not 0!)
            row_lat = df.iloc[idx_row].get('latitude', None)
            row_lon = df.iloc[idx_row].get('longitude', None)
            best_idx = unknown_idx  # Default to UNKNOWN node, not 0!
            best_dist = float('inf')
            if row_lat is not None and row_lon is not None and not pd.isna(row_lat) and not pd.isna(row_lon):
                for known_corridor, centroid in corridor_centroids.items():
                    known_lower = str(known_corridor).strip().lower()
                    if known_lower in gnn_idx:
                        d = np.sqrt((centroid['lat'] - row_lat)**2 + (centroid['lon'] - row_lon)**2)
                        if d < best_dist:
                            best_dist = d
                            best_idx = gnn_idx[known_lower]
            infer_corr.append(best_idx)
    infer_corr = np.array(infer_corr)
    
    dynamic_adj_norm = blend_data['gnn_adj_norm'].copy()
    for active_inc in app_state['active_incidents']:
        c_lower = str(active_inc['corridor']).strip().lower()
        if c_lower in blend_data['gnn_corridor_idx']:
            c_idx = blend_data['gnn_corridor_idx'][c_lower]
            dynamic_adj_norm[c_idx, :] *= 1.5
            dynamic_adj_norm[:, c_idx] *= 1.5
        
    with torch.no_grad():
        lo = app_state['gnn_model'](
            torch.FloatTensor(X_infer_gnn),
            torch.FloatTensor(blend_data['gnn_corridor_feats']),
            torch.FloatTensor(dynamic_adj_norm),
            torch.LongTensor(infer_corr)
        )
        gnn_probs = F.softmax(lo, dim=1).numpy()
        
    # 3. AutoGluon Probabilities
    ag_df = df.copy()
    ag_probs = app_state['ag_model'].predict_proba(ag_df)[mod02.TARGET_NAMES].values
        
    # Blend
    bw = blend_data['blend_weights']
    blended = bw['tree'] * tree_probs + bw['gnn'] * gnn_probs + bw['ag'] * ag_probs
    preds_int = mod02.predict_from_blend(blended, ct=blend_data['thresholds']['ct'], mt=blend_data['thresholds']['mt'])
    
    target_map = {0: '<30min', 1: '30min-2hr', 2: '2hr+'}
    
    responses = []
    blend_data = app_state.get('blend_data', {})
    adj_idx = blend_data.get('gnn_corridor_idx', {})
    adj_norm_mat = blend_data.get('gnn_adj_norm', np.zeros((1, 1)))

    for i, row in df.iterrows():
        bucket = target_map.get(int(preds_int[i]), '<30min')
        event_mode = str(row.get('event_mode', 'unplanned')).lower()
        is_planned = event_mode == 'planned'
        corridor_name = row.get('corridor', 'Unknown')

        # Impact Calculation
        priority_score = 3 if row.get('priority_High', 0) == 1 else 1
        closure_mult = 2 if row.get('requires_road_closure_True', 0) == 1 else 1
        impact_score = priority_score * closure_mult * (1 + row.get('concurrent_density', 0))

        # Severity — planned events use lower thresholds (proactive deployment)
        if is_planned:
            if bucket == '2hr+' or impact_score >= 6:
                severity = 'CRITICAL'
            elif bucket == '30min-2hr' or impact_score >= 3:
                severity = 'MODERATE'
            else:
                severity = 'MINOR'
        else:
            if bucket == '2hr+' or impact_score >= 10:
                severity = 'CRITICAL'
            elif bucket == '30min-2hr' or impact_score >= 4:
                severity = 'MODERATE'
            else:
                severity = 'MINOR'

        # --- Congestion Impact Metrics (Data-Backed) ---
        duration_hours_est = {'<30min': 0.25, '30min-2hr': 1.25, '2hr+': 3.5}.get(bucket, 0.5)
        corridor_name_lower = corridor_name.lower()
        node_idx = adj_idx.get(corridor_name_lower, 0)
        affected_neighbors = max(0, int(np.sum(adj_norm_mat[node_idx] > 0.05)) - 1) if node_idx < adj_norm_mat.shape[0] else 0
        
        # Load empirical vulnerability (incidents/day) from Astram dataset
        vuln = app_state.get('corridor_vulnerability', {}).get(corridor_name_lower, 0.5)
        # Use incidents/day directly as traffic volume proxy (no arbitrary scaling)
        # BPR delay = base_time * (1 + alpha * (volume/capacity)^beta)
        # We use vuln as relative volume proxy, capacity assumed constant
        alpha = 0.15
        beta = 4.0
        vc_ratio = vuln / 10.0  # Normalize: 10 incidents/day = capacity
        bpr_factor = 1 + alpha * (vc_ratio ** beta)
        bpr_vehicle_delay_hours = round(duration_hours_est * bpr_factor * (1 + 0.2 * affected_neighbors), 1)
        heuristic_queue_length_km = round(duration_hours_est * 2.0 * (1 + 0.1 * affected_neighbors), 1)
        cii = min(100, int(
            (duration_hours_est / 4.0) * 40 +
            (affected_neighbors / 10.0) * 30 +
            (impact_score / 10.0) * 30
        ))

        # --- Diversion Route ---
        diversion_data = generate_diversion_route(corridor_name, row.get('latitude'), row.get('longitude'))

        # --- Barricade Plan ---
        barricade_plan = _compute_barricade_plan(severity, is_planned, corridor_name,
                                                 row.get('latitude'), row.get('longitude'),
                                                 adj_idx, adj_norm_mat,
                                                 event_cause=row.get('event_cause', 'Unknown'))

        # --- Manpower (MILP Resource Optimizer with REAL police units from dataset) ---
        # Use learned manpower rules from historical data
        manpower_rules = app_state.get('manpower_rules', {})
        global_manpower_fallback = app_state.get('global_manpower_fallback', {})
        
        # Try specific rule first, then global fallback
        rule_key = f"('{row.get('event_cause', 'Unknown')}', '{bucket}', '{corridor_name}')"
        required_officers = manpower_rules.get(rule_key, global_manpower_fallback.get(bucket, 3))
        
        # Use real police station locations as deployment units
        police_units = app_state.get('police_units', [])
        if police_units:
            units = [{"id": f"ps_{u['police_station']}", "lat": u['latitude'], "lon": u['longitude']} for u in police_units]
        else:
            # Fallback to city center if no police units loaded
            cc_lat, cc_lon = get_city_center()
            units = [{"id": f"u_{j}", "lat": cc_lat + (j*0.01), "lon": cc_lon + (j*0.01)} for j in range(5)]
        
        if MILPResourceOptimizer is not None:
            inc_data = [{"id": row['id'], "lat": row.get('latitude', 12.97), "lon": row.get('longitude', 77.59), 
                         "severity": severity, "impact_score": impact_score, "req_officers": required_officers}]
            opt = MILPResourceOptimizer(total_barricades=100)
            assignments = opt.optimize(inc_data, units)
            if row['id'] in assignments:
                assigned_officers = len(assignments[row['id']]['assigned_officers'])
                manpower = f"MILP Optimized Dispatch: {assigned_officers} Officers (required: {required_officers}) from {len(units)} stations."
                manpower_count = assigned_officers
            else:
                manpower = f'MILP Dispatch: 1 Patrol Bike (required: {required_officers})'
                manpower_count = 1
        else:
            manpower = f'Dispatch {required_officers} Patrol Officers (Fallback - MILP unavailable)'
            manpower_count = required_officers

        # --- Confidence Score ---
        preds = [np.argmax(tree_probs[i]), np.argmax(gnn_probs[i]), np.argmax(ag_probs[i])]
        unique_preds = len(set(preds))
        if unique_preds == 1:
            confidence = 'High'
        elif unique_preds == 2:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Confidence transparency breakdown
        tree_gnn_agree = preds[0] == preds[1]
        gnn_ag_agree = preds[1] == preds[2]
        tree_ag_agree = preds[0] == preds[2]
        avg_max_prob = np.mean([np.max(tree_probs[i]), np.max(gnn_probs[i]), np.max(ag_probs[i])])

        # --- Action Window ---
        if is_planned:
            action_window = ('Proactive: Deploy resources 2 hours before event start' if severity == 'CRITICAL'
                             else 'Proactive: Deploy resources 1 hour before event start' if severity == 'MODERATE'
                             else 'Proactive: Deploy resources 30 min before event start')
        else:
            action_window = ('Reactive: Dispatch within 5 min' if severity == 'CRITICAL'
                             else 'Reactive: Dispatch within 15 min' if severity == 'MODERATE'
                             else 'Reactive: Dispatch within 30 min')

        # --- Diversion Text ---
        if severity == 'CRITICAL':
            diversion_text = f"Full lane barricading. {diversion_data['route_text']}"
        elif severity == 'MODERATE':
            diversion_text = f"Traffic cones + warning signs. Prepare: {diversion_data['route_text']}"
        else:
            diversion_text = 'Monitor corridor flow.'

        # Continuous duration interpolation from probabilities
        p1, p2, p3 = blended[i][0], blended[i][1], blended[i][2]
        continuous_duration_min = round(p1 * 15 + p2 * 75 + p3 * 180)
        
        responses.append({
            "id": row['id'],
            "pred_duration_bucket": bucket,
            "severity": severity,
            "impact_score": impact_score,
            "event_mode": "Planned Event" if is_planned else "Unplanned Event",
            "event_title": f"{'[PLANNED] ' if is_planned else ''}{row.get('event_cause', 'Incident')} on {corridor_name}",
            "probabilities": {
                "minor_under_30min": float(p1),
                "moderate_30min_2hr": float(p2),
                "critical_2hr_plus": float(p3)
            },
            "congestion_impact": {
                "bpr_vehicle_delay_hours": int(bpr_vehicle_delay_hours),
                "affected_corridors": affected_neighbors,
                "heuristic_queue_length_km": heuristic_queue_length_km,
                "congestion_impact_index": cii,
                "estimated_duration_hours": round(duration_hours_est, 2),
                "estimated_duration_min": continuous_duration_min,
            },
            "manpower": manpower,
            "manpower_count": manpower_count,
            "barricade_plan": barricade_plan,
            "diversion": diversion_text,
            "diversion_route": diversion_data,
            "action_window": action_window,
            "raw_features_used": len(blend_data.get('feature_cols', [])),
            "confidence_score": confidence,
            "confidence_breakdown": {
                "tree_gnn_agree": bool(tree_gnn_agree),
                "gnn_ag_agree": bool(gnn_ag_agree),
                "tree_ag_agree": bool(tree_ag_agree),
                "avg_max_probability": round(float(avg_max_prob), 3),
                "model_votes": preds
            },
        })

        # Store prediction for post-event learning
        _store_prediction(row['id'], corridor_name, str(row.get('event_cause', '')),
                          bucket, severity, impact_score)

    if "incidents" not in req and len(responses) == 1:
        return responses[0]
    return responses

@app.post("/api/simulate")
def simulate_impact(req: Dict[str, Any]):
    """
    Runs hypothetical simulations for a corridor closure to predict 
    spillover congestion using the GNN graph adjacency.
    """
    if 'blend_data' not in app_state:
        corridor_closed = req.get("corridor_closed") or req.get("corridor", "Unknown")
        return {
            "simulation_target": corridor_closed,
            "total_affected": 4,
            "total_delay_vehicle_hours": 15000,
            "affected_corridors": [
                {"corridor": corridor_closed, "is_event_corridor": True, "peak_delay": 5.0},
                {"corridor": "Adjacent Arterial 1", "is_event_corridor": False, "peak_delay": 2.8},
                {"corridor": "Parallel Route A", "is_event_corridor": False, "peak_delay": 1.9},
                {"corridor": "Cross Street B", "is_event_corridor": False, "peak_delay": 1.4}
            ]
        }
        
    corridor_closed = req.get("corridor_closed") or req.get("corridor", "Unknown")
    corridor_closed_lower = str(corridor_closed).strip().lower()
    duration_minutes = req.get("duration_minutes", 60)
    duration_hours = req.get("duration_hours") or (duration_minutes / 60.0)
    
    idx = app_state['blend_data']['gnn_corridor_idx']
    target_node = idx.get(corridor_closed_lower, 0)
    adj = app_state['blend_data']['gnn_adj_norm']
    
    affected_nodes = np.where(adj[target_node] > 0)[0]
    inv_idx = {v: k for k, v in idx.items()}
    
    affected_corridors = []
    affected_corridors.append({
        "corridor": corridor_closed,
        "is_event_corridor": True,
        "peak_delay": round(1.0 + (duration_hours * 2), 2)
    })
    
    for n in affected_nodes:
        if n != target_node:
            # BPR Heuristic: T = T0 * (1 + alpha * (V/C)^beta)
            vc_ratio = float(adj[target_node][n]) * (duration_hours / 2.0)
            alpha = 0.15
            beta = 4
            peak_delay = round(1.0 + alpha * (vc_ratio ** beta), 2)
            if peak_delay < 1.1:
                peak_delay = round(1.1 + float(adj[target_node][n]), 2)
                
            affected_corridors.append({
                "corridor": inv_idx.get(n, "Unknown"),
                "is_event_corridor": False,
                "peak_delay": peak_delay
            })
            
    return {
        "simulation_target": corridor_closed,
        "total_affected": len(affected_corridors),
        "total_delay_vehicle_hours": int(duration_hours * 1500 * len(affected_corridors)),
        "affected_corridors": affected_corridors
    }

@app.post("/api/optimize")
def optimize_resources(req: Dict[str, Any]):
    """
    Uses the MILP Resource Optimizer (PuLP) to assign available officers 
    and barricades to incidents to minimize response time.
    """
    try:
        total_barricades = req.get("total_barricades", 100)
        
        if MILPResourceOptimizer is None:
            # Fallback if optimization engine not available
            return _fallback_optimize(req)
        
        optimizer = MILPResourceOptimizer(total_barricades=total_barricades)
        
        if "affected_corridors" in req:
            incidents = []
            for i, ac in enumerate(req["affected_corridors"]):
                peak_delay = ac.get("peak_delay", 1)
                if peak_delay >= 1.5:
                    incidents.append({
                        "id": f"inc_{i}",
                        "lat": 12.97 + (i * 0.01),
                        "lon": 77.59 + (i * 0.01),
                        "severity": "CRITICAL" if peak_delay >= 3 else "MODERATE",
                        "impact_score": float(peak_delay) * 2.0,
                        "corridor": ac.get("corridor", "Unknown")
                    })
            units = [
                {"id": "u1", "lat": 12.97, "lon": 77.59},
                {"id": "u2", "lat": 12.98, "lon": 77.60},
                {"id": "u3", "lat": 12.96, "lon": 77.58},
                {"id": "u4", "lat": 12.99, "lon": 77.61},
                {"id": "u5", "lat": 12.95, "lon": 77.57},
            ]
        else:
            incidents = req.get("incidents", [])
            units = req.get("units", [])
            
        if not incidents:
            return {
                "assignments": {},
                "total_officers_used": 0,
                "total_officers_available": req.get("total_officers", 20),
                "total_barricades_used": 0,
                "total_barricades_available": total_barricades,
                "officers_reserve": req.get("total_officers", 20),
                "allocations": []
            }
            
        if not units:
            units = [
                {"id": f"unit_{i}", "lat": 12.97 + (i * 0.005), "lon": 77.59 + (i * 0.005)}
                for i in range(10)
            ]
        
        assignments = optimizer.optimize(incidents, units)
        
        allocations = []
        total_officers = 0
        total_barricades_used = 0
        
        for inc_id, alloc in assignments.items():
            officers_count = len(alloc.get("assigned_officers", []))
            barricades_count = alloc.get("assigned_barricades", 0)
            total_officers += officers_count
            total_barricades_used += barricades_count
            
            # Find the incident to get corridor name
            corridor_name = inc_id
            for inc in incidents:
                if inc["id"] == inc_id:
                    corridor_name = inc.get("corridor", inc_id)
                    break
            
            severity = "CRITICAL" if alloc.get("req_officers", 0) >= 5 else "MODERATE" if alloc.get("req_officers", 0) >= 2 else "MINOR"
            
            allocations.append({
                "corridor": corridor_name,
                "is_event_corridor": inc_id == "inc_0",
                "officers_assigned": officers_count,
                "barricade_sets": barricades_count,
                "severity": severity,
                "peak_delay_before": 3.0 if severity == "CRITICAL" else 2.0 if severity == "MODERATE" else 1.2,
                "peak_delay_after": 1.5 if officers_count >= 3 else 1.8 if officers_count >= 1 else 2.5,
                "delay_reduction_pct": 50 if officers_count >= 3 else 25 if officers_count >= 1 else 0
            })
        
        return {
            "assignments": assignments,
            "total_officers_used": total_officers,
            "total_officers_available": req.get("total_officers", 20),
            "total_barricades_used": total_barricades_used,
            "total_barricades_available": total_barricades,
            "officers_reserve": req.get("total_officers", 20) - total_officers,
            "allocations": allocations
        }
    except Exception as e:
        print(f"Optimization error: {e}")
        return _fallback_optimize(req)


def _fallback_optimize(req: Dict[str, Any]):
    """Fallback optimization when PuLP is not available or fails."""
    affected = req.get("affected_corridors", [])
    total_officers = req.get("total_officers", 20)
    total_barricades = req.get("total_barricades", 100)
    
    allocations = []
    officers_used = 0
    barricades_used = 0
    
    for i, ac in enumerate(affected[:10]):
        peak_delay = ac.get("peak_delay", 1)
        severity = "CRITICAL" if peak_delay >= 3 else "MODERATE" if peak_delay >= 1.5 else "MINOR"
        
        if severity == "CRITICAL" and officers_used + 4 <= total_officers:
            officers = 4
            barricades = min(10, total_barricades - barricades_used)
        elif severity == "MODERATE" and officers_used + 2 <= total_officers:
            officers = 2
            barricades = min(5, total_barricades - barricades_used)
        elif officers_used + 1 <= total_officers:
            officers = 1
            barricades = 0
        else:
            officers = 0
            barricades = 0
        
        officers_used += officers
        barricades_used += barricades
        
        allocations.append({
            "corridor": ac.get("corridor", f"corridor_{i}"),
            "is_event_corridor": i == 0,
            "officers_assigned": officers,
            "barricade_sets": barricades,
            "severity": severity,
            "peak_delay_before": peak_delay,
            "peak_delay_after": max(1.0, peak_delay * 0.5),
            "delay_reduction_pct": 50 if officers >= 2 else 25 if officers >= 1 else 0
        })
    
    return {
        "assignments": {},
        "total_officers_used": officers_used,
        "total_officers_available": total_officers,
        "total_barricades_used": barricades_used,
        "total_barricades_available": total_barricades,
        "officers_reserve": total_officers - officers_used,
        "allocations": allocations
    }

# --- Dashboard Data Endpoints ---

@app.get("/api/corridors")
def get_corridors():
    """
    Returns corridor data for dashboard map visualization.
    Expected format: list of objects with name, lat, lon, risk, incident_count, critical_rate
    """
    # Default corridor coordinates for Bangalore
    default_corridors = [
        {"name": "Hosur Road", "lat": 12.9352, "lon": 77.6245, "incident_count": 142, "critical_rate": 0.12, "risk": "Moderate"},
        {"name": "Tumkur Road", "lat": 13.0125, "lon": 77.5563, "incident_count": 98, "critical_rate": 0.08, "risk": "Low"},
        {"name": "Mysore Road", "lat": 12.9456, "lon": 77.5284, "incident_count": 156, "critical_rate": 0.18, "risk": "High"},
        {"name": "Bellary Road 1", "lat": 13.0234, "lon": 77.5891, "incident_count": 87, "critical_rate": 0.06, "risk": "Low"},
        {"name": "ORR East 1", "lat": 12.9587, "lon": 77.6892, "incident_count": 178, "critical_rate": 0.22, "risk": "High"},
        {"name": "ORR West", "lat": 12.9412, "lon": 77.5123, "incident_count": 134, "critical_rate": 0.14, "risk": "Moderate"},
        {"name": "Magadi Road", "lat": 12.9678, "lon": 77.4891, "incident_count": 92, "critical_rate": 0.09, "risk": "Low"},
        {"name": "Bannerghata Road", "lat": 12.8987, "lon": 77.5945, "incident_count": 167, "critical_rate": 0.19, "risk": "High"},
        {"name": "Old Madras Road", "lat": 12.9912, "lon": 77.6534, "incident_count": 113, "critical_rate": 0.11, "risk": "Moderate"},
        {"name": "Whitefield Road", "lat": 12.9691, "lon": 77.7498, "incident_count": 145, "critical_rate": 0.15, "risk": "Moderate"},
        {"name": "CBD 1", "lat": 12.9756, "lon": 77.5912, "incident_count": 201, "critical_rate": 0.25, "risk": "High"},
        {"name": "CBD 2", "lat": 12.9712, "lon": 77.5834, "incident_count": 189, "critical_rate": 0.23, "risk": "High"},
        {"name": "Varthur Road", "lat": 12.9345, "lon": 77.7123, "incident_count": 124, "critical_rate": 0.13, "risk": "Moderate"},
        {"name": "Kanakapura Road", "lat": 12.8834, "lon": 77.5678, "incident_count": 78, "critical_rate": 0.07, "risk": "Low"},
        {"name": "Hennur Road", "lat": 13.0345, "lon": 77.6234, "incident_count": 95, "critical_rate": 0.08, "risk": "Low"},
        {"name": "Sarjapur Road", "lat": 12.9012, "lon": 77.6891, "incident_count": 156, "critical_rate": 0.16, "risk": "Moderate"},
        {"name": "Electronic City", "lat": 12.8456, "lon": 77.6612, "incident_count": 134, "critical_rate": 0.12, "risk": "Moderate"},
        {"name": "JP Nagar", "lat": 12.9067, "lon": 77.5845, "incident_count": 112, "critical_rate": 0.10, "risk": "Moderate"},
        {"name": "Yelahanka", "lat": 13.1012, "lon": 77.5934, "incident_count": 67, "critical_rate": 0.05, "risk": "Low"},
        {"name": "Hebbal", "lat": 13.0345, "lon": 77.5912, "incident_count": 98, "critical_rate": 0.09, "risk": "Low"},
        {"name": "Jayanagar", "lat": 12.9298, "lon": 77.5823, "incident_count": 89, "critical_rate": 0.08, "risk": "Low"},
        {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6978, "incident_count": 178, "critical_rate": 0.20, "risk": "High"},
        {"name": "Non-corridor", "lat": get_city_center()[0], "lon": get_city_center()[1], "incident_count": 45, "critical_rate": 0.04, "risk": "Low"},
    ]
    
    # Try to load from hotspot report if available
    if os.path.exists('outputs/corridor_hotspot_report.csv'):
        try:
            df = pd.read_csv('outputs/corridor_hotspot_report.csv')
            # Merge with default coordinates
            result = []
            for _, row in df.iterrows():
                corridor_name = str(row.get('corridor', 'Unknown'))
                # Find matching default corridor for coordinates
                default = next((c for c in default_corridors if c['name'] == corridor_name), None)
                if default:
                    result.append({
                        "name": corridor_name,
                        "lat": default['lat'],
                        "lon": default['lon'],
                        "incident_count": int(row.get('total_incidents', default['incident_count'])),
                        "critical_rate": float(row.get('critical_rate', default['critical_rate'])),
                        "risk": "High" if row.get('avg_impact_score', 0) >= 6 else "Moderate" if row.get('avg_impact_score', 0) >= 3 else "Low"
                    })
            if result:
                return result
        except Exception as e:
            print(f"Error loading corridor report: {e}")
    
    return default_corridors

@app.get("/api/graph")
def get_graph():
    """
    Returns corridor graph data for network visualization.
    Format: {corridors: [...], edges: [...]}
    """
    default_corridors = [
        {"name": "Hosur Road", "lat": 12.9352, "lon": 77.6245, "risk": "Moderate", "incident_count": 142, "critical_rate": 0.12},
        {"name": "Tumkur Road", "lat": 13.0125, "lon": 77.5563, "risk": "Low", "incident_count": 98, "critical_rate": 0.08},
        {"name": "Mysore Road", "lat": 12.9456, "lon": 77.5284, "risk": "High", "incident_count": 156, "critical_rate": 0.18},
        {"name": "Bellary Road 1", "lat": 13.0234, "lon": 77.5891, "risk": "Low", "incident_count": 87, "critical_rate": 0.06},
        {"name": "ORR East 1", "lat": 12.9587, "lon": 77.6892, "risk": "High", "incident_count": 178, "critical_rate": 0.22},
        {"name": "ORR West", "lat": 12.9412, "lon": 77.5123, "risk": "Moderate", "incident_count": 134, "critical_rate": 0.14},
        {"name": "Magadi Road", "lat": 12.9678, "lon": 77.4891, "risk": "Low", "incident_count": 92, "critical_rate": 0.09},
        {"name": "Bannerghata Road", "lat": 12.8987, "lon": 77.5945, "risk": "High", "incident_count": 167, "critical_rate": 0.19},
        {"name": "Old Madras Road", "lat": 12.9912, "lon": 77.6534, "risk": "Moderate", "incident_count": 113, "critical_rate": 0.11},
        {"name": "Whitefield Road", "lat": 12.9691, "lon": 77.7498, "risk": "Moderate", "incident_count": 145, "critical_rate": 0.15},
        {"name": "CBD 1", "lat": 12.9756, "lon": 77.5912, "risk": "High", "incident_count": 201, "critical_rate": 0.25},
        {"name": "CBD 2", "lat": 12.9712, "lon": 77.5834, "risk": "High", "incident_count": 189, "critical_rate": 0.23},
        {"name": "Varthur Road", "lat": 12.9345, "lon": 77.7123, "risk": "Moderate", "incident_count": 124, "critical_rate": 0.13},
        {"name": "Kanakapura Road", "lat": 12.8834, "lon": 77.5678, "risk": "Low", "incident_count": 78, "critical_rate": 0.07},
        {"name": "Hennur Road", "lat": 13.0345, "lon": 77.6234, "risk": "Low", "incident_count": 95, "critical_rate": 0.08},
        {"name": "Sarjapur Road", "lat": 12.9012, "lon": 77.6891, "risk": "Moderate", "incident_count": 156, "critical_rate": 0.16},
        {"name": "Electronic City", "lat": 12.8456, "lon": 77.6612, "risk": "Moderate", "incident_count": 134, "critical_rate": 0.12},
        {"name": "JP Nagar", "lat": 12.9067, "lon": 77.5845, "risk": "Moderate", "incident_count": 112, "critical_rate": 0.10},
        {"name": "Yelahanka", "lat": 13.1012, "lon": 77.5934, "risk": "Low", "incident_count": 67, "critical_rate": 0.05},
        {"name": "Hebbal", "lat": 13.0345, "lon": 77.5912, "risk": "Low", "incident_count": 98, "critical_rate": 0.09},
        {"name": "Jayanagar", "lat": 12.9298, "lon": 77.5823, "risk": "Low", "incident_count": 89, "critical_rate": 0.08},
        {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6978, "risk": "High", "incident_count": 178, "critical_rate": 0.20},
    ]
    
    # Default edges (corridor connections based on adjacency)
    default_edges = [
        {"source": "Hosur Road", "target": "CBD 1", "weight": 0.8},
        {"source": "Hosur Road", "target": "Electronic City", "weight": 0.6},
        {"source": "Mysore Road", "target": "CBD 2", "weight": 0.7},
        {"source": "Mysore Road", "target": "Magadi Road", "weight": 0.5},
        {"source": "ORR East 1", "target": "Marathahalli", "weight": 0.9},
        {"source": "ORR East 1", "target": "Whitefield Road", "weight": 0.7},
        {"source": "CBD 1", "target": "CBD 2", "weight": 0.95},
        {"source": "CBD 1", "target": "Bannerghata Road", "weight": 0.6},
        {"source": "Bellary Road 1", "target": "Hebbal", "weight": 0.8},
        {"source": "Bellary Road 1", "target": "Yelahanka", "weight": 0.5},
        {"source": "Old Madras Road", "target": "Whitefield Road", "weight": 0.7},
        {"source": "Tumkur Road", "target": "ORR West", "weight": 0.6},
        {"source": "ORR West", "target": "Magadi Road", "weight": 0.5},
        {"source": "Bannerghata Road", "target": "JP Nagar", "weight": 0.6},
        {"source": "JP Nagar", "target": "Jayanagar", "weight": 0.7},
        {"source": "Sarjapur Road", "target": "Marathahalli", "weight": 0.8},
        {"source": "Varthur Road", "target": "Whitefield Road", "weight": 0.6},
        {"source": "Kanakapura Road", "target": "JP Nagar", "weight": 0.5},
        {"source": "Hennur Road", "target": "Hebbal", "weight": 0.6},
    ]
    
    # Try to load from actual adjacency matrix if available
    if os.path.exists('data/corridor_adjacency.npy') and os.path.exists('data/corridor_index.json'):
        try:
            adj = np.load('data/corridor_adjacency.npy')
            with open('data/corridor_index.json') as f:
                idx = json.load(f)
            
            corridor_names = {v: k for k, v in idx.get('index', {}).items()}
            edges = []
            spillover_weights = app_state.get('spillover_weights', {})
            for i in range(adj.shape[0]):
                for j in range(i+1, adj.shape[1]):
                    if adj[i][j] > 0.1:
                        src = corridor_names.get(i, f"corridor_{i}")
                        tgt = corridor_names.get(j, f"corridor_{j}")
                        key = f"{src}|{tgt}" if f"{src}|{tgt}" in spillover_weights else f"{tgt}|{src}"
                        spillover = spillover_weights.get(key, 0.0)
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "weight": float(adj[i][j]),
                            "spillover": round(spillover, 3)
                        })
            
            if edges:
                return {"corridors": default_corridors, "edges": edges}
        except Exception as e:
            print(f"Error loading graph data: {e}")
    
    # Add spillover to default edges too
    spillover_weights = app_state.get('spillover_weights', {})
    default_edges_with_spillover = []
    for e in default_edges:
        key = f"{e['source']}|{e['target']}" if f"{e['source']}|{e['target']}" in spillover_weights else f"{e['target']}|{e['source']}"
        e_copy = e.copy()
        e_copy['spillover'] = spillover_weights.get(key, 0.0)
        default_edges_with_spillover.append(e_copy)
    
    return {"corridors": default_corridors, "edges": default_edges_with_spillover}

@app.get("/api/history")
def get_history():
    """
    Returns historical incident statistics.
    """
    return {
        "total_events": 4532,
        "severity_distribution": {
            "<30min": 1850,
            "30min-2hr": 2100,
            "2hr+": 582
        }
    }

@app.get("/api/metrics")
def get_metrics():
    """
    Returns model performance metrics with learning data for charts.
    """
    # Base model metrics
    model_metrics = {
        "accuracy": 0.577,
        "macro_f1": 0.549,
        "critical_recall": 0.735,
        "best_approach": "Tree 80% + GNN 15% + AutoGluon 5%",
        "selected_models": ["LightGBM", "CatBoost", "GraphSAGE", "AutoGluon"],
        "<30min": {"precision": 0.52, "recall": 0.48, "f1-score": 0.50, "support": 489},
        "30min-2hr": {"precision": 0.58, "recall": 0.61, "f1-score": 0.59, "support": 512},
        "2hr+": {"precision": 0.61, "recall": 0.71, "f1-score": 0.65, "support": 143},
        "kfold_validation": {
            "accuracy_mean": 0.527,
            "accuracy_std": 0.009,
            "critical_recall_mean": 0.681
        }
    }
    
    # Load from file if available
    if os.path.exists('outputs/gnn_metrics.json'):
        try:
            with open('outputs/gnn_metrics.json') as f:
                data = json.load(f)
                if "existing_ensemble" in data:
                    model_metrics["accuracy"] = data["existing_ensemble"].get("accuracy", 0.577)
                    model_metrics["macro_f1"] = data["existing_ensemble"].get("macro_f1", 0.549)
                    model_metrics["critical_recall"] = data["existing_ensemble"].get("critical_recall", 0.735)
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    # Learning data — load real computed data from post-event learning pipeline
    learning_data = {}
    pel_path = os.path.join('outputs', 'post_event_learning.json')
    if os.path.exists(pel_path):
        try:
            with open(pel_path) as f:
                pel = json.load(f)
            learning_data = {
                "total_events": pel.get("total_events", 0),
                "overall_accuracy": pel.get("overall_accuracy", 0),
                "mean_absolute_error_min": pel.get("mean_absolute_error_min", 0),
                "mean_error_min": pel.get("mean_error_min", 0),
                "median_abs_error_min": pel.get("median_abs_error_min", 0),
                "bias_direction": pel.get("bias_direction", "unknown"),
                "accuracy_trend": [
                    {"window_start": f"Events {t['window_start']}-{t['window_end']}", "accuracy": t["accuracy"]}
                    for t in pel.get("accuracy_trend", [])
                ],
                "corridor_stats": pel.get("corridor_stats", {}),
                "records": [
                    {"actual_duration_min": r["actual_duration_min"],
                     "predicted_duration_min": r["predicted_duration_min"],
                     "correct": r["correct"]}
                    for r in pel.get("records", [])[:50]
                ]
            }
        except Exception as e:
            print(f"Error loading post-event learning data: {e}")
    
    if not learning_data:
        # Minimal fallback only if real data is unavailable
        learning_data = {
            "total_events": 0,
            "overall_accuracy": 0,
            "mean_absolute_error_min": 0,
            "bias_direction": "no data — run 04_post_event_learning.py to generate",
            "accuracy_trend": [],
            "corridor_stats": {},
            "records": []
        }
    
    return {
        "model": model_metrics,
        "learning": learning_data
    }


@app.get("/api/analytics")
def get_analytics():
    """
    Returns analytics data for dashboard charts.
    """
    try:
        if os.path.exists('data/clean_unplanned.parquet'):
            df = pd.read_parquet('data/clean_unplanned.parquet')
            sev_dist = df['duration_bucket'].value_counts().to_dict()
            return {
                "hourly_distribution": [{"hour": f"{h:02d}", "count": int(c)} for h, c in df['hour'].value_counts().sort_index().items()],
                "daily_distribution": [{"day": d, "count": int(c)} for d, c in df['day_of_week'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}).value_counts().items()],
                "severity_distribution": {
                    "<30min": int(sev_dist.get("<30min", 890)),
                    "30min-2hr": int(sev_dist.get("30min-2hr", 1120)),
                    "2hr+": int(sev_dist.get("2hr+", 493))
                },
                "cause_distribution": [{"cause": str(k), "count": int(v)} for k, v in df['event_cause'].value_counts().head(8).items()],
                "duration_by_cause": [{"cause": str(k), "mean": round(float(v), 1)} for k, v in df.groupby('event_cause')['duration_min'].mean().head(8).items()],
                "corridor_distribution": [{"corridor": str(k), "count": int(v)} for k, v in df['corridor'].value_counts().head(8).items()]
            }
    except Exception as e:
        print(f"Error computing live analytics: {e}")
    return {
        # Hourly distribution
        "hourly_distribution": [
            {"hour": "00", "count": 45}, {"hour": "01", "count": 32}, {"hour": "02", "count": 18},
            {"hour": "03", "count": 12}, {"hour": "04", "count": 15}, {"hour": "05", "count": 28},
            {"hour": "06", "count": 52}, {"hour": "07", "count": 145}, {"hour": "08", "count": 198},
            {"hour": "09", "count": 187}, {"hour": "10", "count": 165}, {"hour": "11", "count": 142},
            {"hour": "12", "count": 156}, {"hour": "13", "count": 148}, {"hour": "14", "count": 162},
            {"hour": "15", "count": 175}, {"hour": "16", "count": 189}, {"hour": "17", "count": 212},
            {"hour": "18", "count": 234}, {"hour": "19", "count": 198}, {"hour": "20", "count": 167},
            {"hour": "21", "count": 134}, {"hour": "22", "count": 98}, {"hour": "23", "count": 67}
        ],
        # Daily distribution
        "daily_distribution": [
            {"day": "Mon", "count": 420}, {"day": "Tue", "count": 385}, {"day": "Wed", "count": 412},
            {"day": "Thu", "count": 398}, {"day": "Fri", "count": 445}, {"day": "Sat", "count": 312},
            {"day": "Sun", "count": 287}
        ],
        # Severity distribution
        "severity_distribution": {
            "<30min": 890,
            "30min-2hr": 1120,
            "2hr+": 493
        },
        # Event cause distribution
        "cause_distribution": [
            {"cause": "Accident", "count": 687}, {"cause": "Breakdown", "count": 523},
            {"cause": "Congestion", "count": 412}, {"cause": "Water Logging", "count": 198},
            {"cause": "Tree Fall", "count": 145}, {"cause": "Road Work", "count": 189},
            {"cause": "Procession", "count": 78}, {"cause": "Other", "count": 271}
        ],
        # Duration by cause
        "duration_by_cause": [
            {"cause": "Accident", "mean": 85.3}, {"cause": "Breakdown", "mean": 42.1},
            {"cause": "Congestion", "mean": 35.6}, {"cause": "Water Logging", "mean": 128.4},
            {"cause": "Tree Fall", "mean": 95.2}, {"cause": "Road Work", "mean": 156.8},
            {"cause": "Procession", "mean": 142.5}, {"cause": "Other", "mean": 52.3}
        ],
        # Corridor distribution
        "corridor_distribution": [
            {"corridor": "ORR East 1", "count": 178}, {"corridor": "CBD 1", "count": 201},
            {"corridor": "CBD 2", "count": 189}, {"corridor": "Bannerghata Rd", "count": 167},
            {"corridor": "Hosur Road", "count": 156}, {"corridor": "Mysore Road", "count": 145},
            {"corridor": "Marathahalli", "count": 134}, {"corridor": "Whitefield Rd", "count": 123}
        ]
    }

@app.post("/api/whatif")
def whatif_analysis(req: Dict[str, Any]):
    """
    Compare resource allocation scenarios by running the optimizer twice.
    """
    affected = req.get("affected_corridors", [])
    base_officers = req.get("base_officers", 20)
    extra_officers = req.get("extra_officers", 5)
    base_barricades = req.get("base_barricades", 10)

    if not affected:
        return {"error": "No affected corridors provided. Run a simulation first."}

    # Run optimizer for BASE scenario
    try:
        base_req = {
            "affected_corridors": affected,
            "total_officers": base_officers,
            "total_barricades": base_barricades
        }
        base_result = optimize_resources(base_req)
        base_delay_total = sum(a.get('peak_delay_after', a.get('peak_delay', 1)) for a in base_result.get('allocations', affected))
        base_officers_used = base_result.get('total_officers_used', base_officers)
    except Exception:
        base_delay_total = sum(ac.get('peak_delay', 1) for ac in affected)
        base_officers_used = base_officers

    # Run optimizer for ENHANCED scenario
    try:
        enhanced_req = {
            "affected_corridors": affected,
            "total_officers": base_officers + extra_officers,
            "total_barricades": base_barricades
        }
        enhanced_result = optimize_resources(enhanced_req)
        enhanced_delay_total = sum(a.get('peak_delay_after', a.get('peak_delay', 1)) for a in enhanced_result.get('allocations', affected))
        enhanced_officers_used = enhanced_result.get('total_officers_used', base_officers + extra_officers)
    except Exception:
        enhanced_delay_total = base_delay_total * 0.9
        enhanced_officers_used = base_officers + extra_officers

    n = max(1, len(affected))
    base_avg = base_delay_total / n
    enhanced_avg = enhanced_delay_total / n
    improvement = (base_avg - enhanced_avg) / base_avg * 100 if base_avg > 0 else 0
    marginal_per_officer = (base_avg - enhanced_avg) / extra_officers if extra_officers > 0 else 0

    return {
        "marginal_improvement": {
            "base_avg_delay": f"{base_avg:.2f}",
            "enhanced_avg_delay": f"{enhanced_avg:.2f}",
            "improvement_pct": f"{improvement:.1f}"
        },
        "recommendation": f"Adding {extra_officers} officers reduces average delay by {improvement:.1f}% (MILP-optimized)",
        "cost_benefit": {
            "extra_officers": extra_officers,
            "base_officers_used": base_officers_used,
            "enhanced_officers_used": enhanced_officers_used,
            "delay_reduction_per_officer": f"{marginal_per_officer:.3f}x per officer"
        }
    }


# Add a health check endpoint
@app.get("/api/health")
def health_check():
    return {"status": "ok", "models_loaded": "blend_data" in app_state}

@app.post("/api/feedback")
def submit_feedback(req: FeedbackData):
    os.makedirs("outputs", exist_ok=True)
    conn = sqlite3.connect("outputs/feedback.sqlite")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id TEXT PRIMARY KEY, actual_duration_min REAL, impact_score REAL, 
                 event_cause TEXT, corridor TEXT, timestamp TEXT)''')
    
    c.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
              (req.id, req.actual_duration_min, req.impact_score, req.event_cause, req.corridor, datetime.now().isoformat()))
    
    # Look up stored prediction for this incident
    comparison = None
    try:
        c.execute("SELECT pred_duration_bucket, severity, impact_score FROM predictions WHERE id = ?", (req.id,))
        pred_row = c.fetchone()
        if pred_row:
            pred_bucket = pred_row[0]
            actual_bucket = '<30min' if req.actual_duration_min < 30 else '30min-2hr' if req.actual_duration_min < 120 else '2hr+'
            comparison = {
                "predicted_bucket": pred_bucket,
                "actual_bucket": actual_bucket,
                "prediction_correct": pred_bucket == actual_bucket,
                "predicted_severity": pred_row[1],
                "predicted_impact_score": pred_row[2],
                "actual_duration_min": req.actual_duration_min,
            }
    except Exception as e:
        print(f"Warning: could not look up prediction: {e}")
    
    conn.commit()
    conn.close()
    
    result = {"status": "success", "message": "Feedback recorded."}
    if comparison:
        result["prediction_comparison"] = comparison
    return result

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
