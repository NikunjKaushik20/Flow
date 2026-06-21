import pandas as pd
import numpy as np
import pickle
import json
import logging
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from autogluon.tabular import TabularPredictor

import importlib
import __main__
mod02 = importlib.import_module("02_model_training")
prepare_frame = mod02.prepare_frame
proba = mod02.proba
predict_from_blend = mod02.predict_from_blend
TARGET_NAMES = mod02.TARGET_NAMES

setattr(__main__, 'text_col_selector', mod02.text_col_selector)
setattr(__main__, 'TargetStatsEncoder', mod02.TargetStatsEncoder)
setattr(__main__, 'to_dense_matrix', mod02.to_dense_matrix)
from utils_gnn import GridlockGNN


PLANNED_CAUSES = {'procession', 'protest', 'construction'}
UNPLANNED_CAUSES = {'accident', 'vehicle_breakdown', 'tree_fall', 'water_logging',
                    'congestion', 'road_conditions', 'pot_holes', 'others'}

def get_event_mode(row):
    cause = str(row.get('event_cause', '')).lower()
    return 'planned' if cause in PLANNED_CAUSES else 'unplanned'


def calculate_impact_score(row):
    priority_score = 1
    if row.get('priority_High', 0) == 1:
        priority_score = 3
    elif row.get('priority_Medium', 0) == 1:
        priority_score = 2
    closure_mult = 2 if row.get('requires_road_closure_True', 0) == 1 else 1
    density = row.get('concurrent_density', 0)
    return priority_score * closure_mult * (1 + density)


class ResourceOptimizer:
    """
    Allocates limited city resources (officers, barricades) dynamically 
    to concurrent incidents based on severity and impact score.
    """
    def __init__(self, total_officers=50, total_barricades=100):
        self.total_officers = total_officers
        self.total_barricades = total_barricades

    def optimize(self, df):
        df_out = df.copy()
        
        # Initialize output columns
        for col in ['Severity', 'Mode', 'Manpower', 'Barricading', 'Diversion', 'Action_Window']:
            if col not in df_out.columns:
                df_out[col] = ''
        
        officers_left = self.total_officers
        barricades_left = self.total_barricades
        
        # Sort by impact score descending for greedy optimal allocation
        sorted_indices = df_out.sort_values(by='impact_score', ascending=False).index
        
        for idx in sorted_indices:
            row = df_out.loc[idx]
            layer1_flag = row.get('layer1_critical_flag', 0)
            pred_duration = row.get('pred_duration_bucket', '<30min')
            impact_score = row.get('impact_score', 0)
            event_mode = row.get('event_mode', 'unplanned')
            
            is_critical = layer1_flag == 1 or pred_duration == '2hr+' or impact_score >= 10
            is_moderate = pred_duration == '30min-2hr' or impact_score >= 4
            
            pol = {
                'Severity': 'MINOR', 
                'Mode': 'Planned Event' if event_mode == 'planned' else 'Unplanned Event',
                'Diversion': 'None',
                'Action_Window': 'Reactive — Dispatch within 30 min'
            }
            
            # 1. Determine Ideal Requirements
            if is_critical:
                ideal_officers = 8 if event_mode == 'planned' else 5
                ideal_barricades = 25 if event_mode == 'planned' else 20
                pol['Severity'] = 'CRITICAL'
                pol['Diversion'] = 'Trigger Immediate Traffic Diversion Protocol'
                pol['Action_Window'] = 'Proactive — Deploy 2 hours before event start' if event_mode == 'planned' else 'Reactive — Dispatch within 5 min'
            elif is_moderate:
                ideal_officers = 4 if event_mode == 'planned' else 2
                ideal_barricades = 15 if event_mode == 'planned' else 10
                pol['Severity'] = 'MODERATE'
                pol['Diversion'] = 'Prepare Alternate Route — Activate if needed'
                pol['Action_Window'] = 'Proactive — Deploy 1 hour before event start' if event_mode == 'planned' else 'Reactive — Dispatch within 15 min'
            else:
                ideal_officers = 2 if event_mode == 'planned' else 1
                ideal_barricades = 5 if event_mode == 'planned' else 0
                pol['Severity'] = 'MINOR'
                pol['Action_Window'] = 'Proactive — Deploy 30 min before event start' if event_mode == 'planned' else 'Reactive — Dispatch within 30 min'

            # 2. Allocate Under Constraints
            allocated_officers = min(ideal_officers, officers_left)
            officers_left -= allocated_officers
            
            allocated_barricades = min(ideal_barricades, barricades_left)
            barricades_left -= allocated_barricades
            
            # 3. Format Dynamic Output
            if allocated_officers >= 5:
                pol['Manpower'] = 'Dispatch 4 Officers + 1 Sub-Inspector'
            elif allocated_officers >= 2:
                pol['Manpower'] = f'Dispatch {allocated_officers} Patrol Officers'
            elif allocated_officers == 1:
                pol['Manpower'] = 'Dispatch 1 Patrol Bike'
            else:
                pol['Manpower'] = 'NO OFFICERS AVAILABLE - Alert City Command'
                
            if allocated_barricades >= 20:
                pol['Barricading'] = 'Full Lane Barricading (20+ units)'
            elif allocated_barricades >= 10:
                pol['Barricading'] = f'Deploy Traffic Cones / Warning Signs ({allocated_barricades} units)'
            elif ideal_barricades > 0 and allocated_barricades == 0:
                pol['Barricading'] = 'NO BARRICADES AVAILABLE'
            else:
                pol['Barricading'] = 'None Required'
                
            for k, v in pol.items():
                df_out.at[idx, k] = v
                
        logging.info(f"Optimization complete. Remaining Resources: {officers_left} Officers, {barricades_left} Barricades.")
        return df_out





def run_layer3_simulation():
    logging.info("Loading 85+10+5 blend model artifacts...")
    try:
        with open('models/layer2_blend.pkl', 'rb') as f:
            blend_data = pickle.load(f)
            
        tree_entries = blend_data['tree_entries']
        tree_names = blend_data['tree_names']
        gnn_scaler = blend_data['gnn_scaler']
        gnn_corridor_feats = blend_data['gnn_corridor_feats']
        gnn_adj_norm = blend_data['gnn_adj_norm']
        gnn_corridor_idx = blend_data['gnn_corridor_idx']
        gnn_available_feats = blend_data['gnn_available_feats']
        blend_weights = blend_data['blend_weights']
        thresholds = blend_data['thresholds']
        
    except FileNotFoundError as e:
        logging.error(f"Model artifacts not found: {e}. Run 02_model_training.py first.")
        return

    # Load corridor name mapping (FIXED PATH)
    corridor_name_map = {}
    try:
        with open('outputs/feature_metadata.json', 'r') as f:
            meta = json.load(f)
        corridors = meta.get('high_cardinality_mappings', {}).get('corridor', [])
        corridor_name_map = {i: name for i, name in enumerate(corridors)}
    except Exception:
        logging.warning("outputs/feature_metadata.json not found, using raw IDs.")

    # Load both datasets via prepare_frame but keeping all events
    logging.info("Preparing data with full feature engineering pipeline...")
    df_full, _ = prepare_frame(unplanned_only=False)
    
    # Layer 1 logic
    is_high_priority = (df_full.get('priority_High', pd.Series(0, index=df_full.index)) == 1)
    requires_closure = (df_full.get('requires_road_closure_True', pd.Series(0, index=df_full.index)) == 1)
    df_full['layer1_critical_flag'] = (is_high_priority & requires_closure).astype(int)
    df_full['event_mode'] = df_full.apply(get_event_mode, axis=1)
    
    # Sample 50 unplanned, and all planned
    planned_df = df_full[df_full['event_mode'] == 'planned']
    unplanned_df = df_full[df_full['event_mode'] == 'unplanned'].sample(min(50, len(df_full[df_full['event_mode'] == 'unplanned'])), random_state=42)
    sample_df = pd.concat([planned_df, unplanned_df])
    
    logging.info(f"Sample size: {len(sample_df)} events")

    # ---- 1. Tree Probabilities ----
    tree_probas = {e["name"]: proba(e, sample_df) for e in tree_entries}
    w = 1.0 / len(tree_names)
    tree_probs = sum(w * tree_probas[n] for n in tree_names)
    
    # ---- 2. GNN Probabilities ----
    for f in gnn_available_feats:
        if f not in sample_df.columns:
            sample_df[f] = 0
    X_infer_gnn = gnn_scaler.transform(sample_df[gnn_available_feats].fillna(0).values)
    infer_corr = np.array([gnn_corridor_idx.get(c, 0) for c in sample_df["corridor"]])
    
    gnn_model = GridlockGNN(len(gnn_available_feats), gnn_corridor_feats.shape[1], len(gnn_corridor_idx), 64, 3, 0.3)
    gnn_model.load_state_dict(torch.load("models/gnn_blend_weights.pt"))
    gnn_model.eval()
    
    with torch.no_grad():
        lo = gnn_model(
            torch.FloatTensor(X_infer_gnn),
            torch.FloatTensor(gnn_corridor_feats),
            torch.FloatTensor(gnn_adj_norm),
            torch.LongTensor(infer_corr)
        )
        gnn_probs = F.softmax(lo, dim=1).numpy()
        
    # ---- 3. AutoGluon Probabilities ----
    ag_probs = np.zeros_like(tree_probs)
    try:
        predictor = TabularPredictor.load("models/autogluon_gridlock", verbosity=0)
        ag_feature_cols = [
            "hour", "day_of_week", "month", "is_weekend", "is_night",
            "minutes_since_midnight", "hour_sin", "hour_cos", "day_sin", "day_cos",
            "month_sin", "month_cos", "created_hour", "created_day_of_week",
            "report_delay_min", "description", "desc_word_count", "desc_char_count",
            "desc_urgency_score", "kw_fire", "kw_spill", "kw_overturned", "kw_injury",
            "kw_block", "kw_heavy", "kw_normal", "kw_tow", "kw_breakdown",
            "latitude_num", "longitude_num", "dist_from_center",
            "has_end_coord",
            "closure_flag", "priority_weight", "rush_hour_flag",
            "rush_hour_x_closure", "high_priority_x_closure",
            "event_cause", "requires_road_closure", "veh_type", "corridor",
            "priority", "police_station", "zone", "junction", "corridor_cause",
        ]
        ag_avail = [c for c in ag_feature_cols if c in sample_df.columns]
        ag_test = sample_df[ag_avail].copy()
        ag_probs_df = predictor.predict_proba(ag_test)
        ag_probs = ag_probs_df[TARGET_NAMES].values
    except Exception as e:
        logging.warning(f"AutoGluon inference failed: {e}")

    # ---- Blend and Predict ----
    blended = blend_weights['tree'] * tree_probs + blend_weights['gnn'] * gnn_probs + blend_weights['ag'] * ag_probs
    preds_int = predict_from_blend(blended, ct=thresholds['ct'], mt=thresholds['mt'])
    
    target_map = {0: '<30min', 1: '30min-2hr', 2: '2hr+'}
    sample_df['pred_duration_bucket'] = [target_map.get(int(p), '<30min') for p in preds_int]

    # Impact score & Layer 3
    sample_df['impact_score'] = sample_df.apply(calculate_impact_score, axis=1)
    
    # Run dynamic resource optimizer
    optimizer = ResourceOptimizer(total_officers=50, total_barricades=100)
    final_output = optimizer.optimize(sample_df)

    # Save
    final_output.to_csv('outputs/layer3_recommendations_sample.csv', index=False)
    print("\nSaved robust policy recommendations to outputs/layer3_recommendations_sample.csv")

    # Per-mode summaries
    planned_out = final_output[final_output['event_mode'] == 'planned']
    unplanned_out = final_output[final_output['event_mode'] == 'unplanned']

    print("\n=== PLANNED EVENTS — Impact Distribution ===")
    print(planned_out['Severity'].value_counts().to_string())

    print("\n=== UNPLANNED EVENTS — Impact Distribution ===")
    print(unplanned_out['Severity'].value_counts().to_string())

    # Corridor hotspot with names
    if 'corridor' in final_output.columns:
        hotspots = final_output.groupby('corridor').agg(
            total_incidents=('pred_duration_bucket', 'count'),
            avg_impact_score=('impact_score', 'mean'),
            critical_incidents=('Severity', lambda x: (x == 'CRITICAL').sum()),
            planned_events=('event_mode', lambda x: (x == 'planned').sum()),
        ).sort_values('avg_impact_score', ascending=False)
        hotspots.to_csv('outputs/corridor_hotspot_report.csv')
        print("\n=== Top 5 Critical Hotspot Corridors ===")
        print(hotspots.head(5).to_string())

    # Sample recommendations
    cols = ['event_mode', 'pred_duration_bucket', 'impact_score', 'Severity', 'Manpower', 'Diversion', 'Action_Window']
    available = [c for c in cols if c in final_output.columns]
    print("\n=== Sample Recommendations (Planned Events) ===")
    print(planned_out[available].to_string())
    print("\n=== Sample Recommendations (Unplanned Events) ===")
    print(unplanned_out[available].head(5).to_string())


if __name__ == "__main__":
    run_layer3_simulation()
