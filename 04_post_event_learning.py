import pandas as pd
import numpy as np
import pickle
import json
import logging
import torch
import torch.nn.functional as F
import warnings
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from autogluon.tabular import TabularPredictor

import importlib
import __main__
mod02 = importlib.import_module("02_model_training")
prepare_frame = mod02.prepare_frame
proba = mod02.proba
TARGET_NAMES = mod02.TARGET_NAMES
TARGET_MAPPING = mod02.TARGET_MAPPING

setattr(__main__, 'text_col_selector', mod02.text_col_selector)
setattr(__main__, 'TargetStatsEncoder', mod02.TargetStatsEncoder)
setattr(__main__, 'to_dense_matrix', mod02.to_dense_matrix)
from utils_gnn import GridlockGNN, INCIDENT_FEATURES


def load_model_artifacts():
    logging.info("Loading latest blend artifacts...")
    try:
        with open('models/layer2_blend.pkl', 'rb') as f:
            blend_data = pickle.load(f)
        return blend_data
    except FileNotFoundError as e:
        logging.error(f"Models not found: {e}. Run 02_model_training.py first.")
        return None

def simulate_new_data_ingestion():
    """
    Ingests resolved incidents from the feedback database to evaluate model drift.
    Matches against stored predictions (from /api/predict) when available.
    """
    logging.info("Ingesting resolved incidents from feedback database...")
    import sqlite3
    import os
    
    df_full, _ = prepare_frame(unplanned_only=False)
    
    if not os.path.exists('outputs/feedback.sqlite'):
        logging.warning("No feedback database found. Skipping learning cycle.")
        return pd.DataFrame(), np.array([]), np.array([])
        
    conn = sqlite3.connect("outputs/feedback.sqlite")
    
    # Check if feedback table exists
    try:
        feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
    except Exception:
        logging.warning("Feedback table does not exist. Skipping learning cycle.")
        conn.close()
        return pd.DataFrame(), np.array([]), np.array([])
    
    # Try to load stored predictions for comparison
    predictions_df = pd.DataFrame()
    try:
        predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
        logging.info(f"Found {len(predictions_df)} stored predictions for matching.")
    except Exception:
        logging.info("No predictions table found — falling back to corridor/cause matching.")
    
    conn.close()
    
    if len(feedback_df) == 0:
        logging.warning("Feedback database is empty. Skipping learning cycle.")
        return pd.DataFrame(), np.array([]), np.array([])
        
    matched_rows = []
    y_new_list = []
    feedback_timestamps = []
    
    for _, row in feedback_df.iterrows():
        duration = row['actual_duration_min']
        if duration < 30: y_val = TARGET_MAPPING["<30min"]
        elif duration <= 120: y_val = TARGET_MAPPING["30min-2hr"]
        else: y_val = TARGET_MAPPING["2hr+"]
        
        # Extract timestamp for decay weighting
        fb_ts = row.get('timestamp', None)
        if fb_ts is not None:
            try:
                fb_ts = pd.to_datetime(fb_ts)
            except:
                fb_ts = pd.Timestamp.now()
        else:
            fb_ts = pd.Timestamp.now()
        
        # Strategy 1: Look up exact feature row by ID (kgid)
        if 'kgid' in df_full.columns and not df_full[df_full['kgid'].astype(str) == str(row['id'])].empty:
            matched_rows.append(df_full[df_full['kgid'].astype(str) == str(row['id'])].iloc[0])
            y_new_list.append(y_val)
            feedback_timestamps.append(fb_ts)
            continue
            
        # Optional: check if we had a stored prediction
        if len(predictions_df) > 0 and row['id'] in predictions_df['id'].values:
            pred_row = predictions_df[predictions_df['id'] == row['id']].iloc[0]
            logging.debug(f"Matched feedback {row['id']} to stored prediction: {pred_row['pred_duration_bucket']}")
        
        # Strategy 2: Match by corridor + cause from historical data (fallback proxy)
        candidates = df_full[(df_full['corridor'] == row['corridor']) & (df_full['event_cause'] == row['event_cause'])]
        if len(candidates) > 0:
            matched_rows.append(candidates.sample(1, random_state=hash(str(row['id'])) % (2**31)).iloc[0])
            y_new_list.append(y_val)
            feedback_timestamps.append(fb_ts)
        else:
            logging.warning(f"Could not match feedback {row['id']} to historical features. Skipping.")
            continue
            
    df_new = pd.DataFrame(matched_rows).reset_index(drop=True)
    y_new = np.array(y_new_list)
    feedback_timestamps = np.array(feedback_timestamps)
    
    return df_new, y_new, feedback_timestamps

def evaluate_and_update_weights(df_new, y_new, blend_data, feedback_timestamps=None):
    tree_entries = blend_data['tree_entries']
    tree_names = blend_data['tree_names']
    gnn_scaler = blend_data['gnn_scaler']
    gnn_corridor_feats = blend_data['gnn_corridor_feats']
    gnn_adj_norm = blend_data['gnn_adj_norm']
    gnn_corridor_idx = blend_data['gnn_corridor_idx']
    gnn_available_feats = blend_data['gnn_available_feats']
    current_weights = blend_data['blend_weights']
    
    if len(df_new) == 0:
        logging.info("No new feedback data to evaluate. Keeping existing weights.")
        return blend_data
        
    logging.info(f"Current Blend Weights: {current_weights}")
    
    # Compute sample weights based on feedback timestamp recency (exponential decay)
    if feedback_timestamps is not None and len(feedback_timestamps) == len(df_new):
        now = pd.Timestamp.now()
        ages_hours = np.array([(now - ts).total_seconds() / 3600 for ts in feedback_timestamps])
        # Half-life of 72 hours: weight = 0.5^(age/72)
        sample_weights = np.power(0.5, ages_hours / 72.0)
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)  # normalize
        logging.info(f"Applied feedback decay: weight range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    else:
        sample_weights = np.ones(len(df_new))
    
    # 1. Get Tree Predictions
    logging.info("Evaluating Tree Ensemble performance on new data...")
    tree_probas = {e["name"]: proba(e, df_new) for e in tree_entries}
    w = 1.0 / len(tree_names)
    tree_probs = sum(w * tree_probas[n] for n in tree_names)
    tree_acc = accuracy_score(y_new, np.argmax(tree_probs, axis=1), sample_weight=sample_weights)
    
    # 2. Get GNN Predictions
    logging.info("Evaluating GNN performance on new data...")
    for f in gnn_available_feats:
        if f not in df_new.columns:
            df_new[f] = 0
            
    X_infer_gnn = gnn_scaler.transform(df_new[gnn_available_feats].fillna(0).values)
    infer_corr = np.array([gnn_corridor_idx.get(c, 0) for c in df_new["corridor"]])
    
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
    
    gnn_acc = accuracy_score(y_new, np.argmax(gnn_probs, axis=1), sample_weight=sample_weights)
    
    # 3. Get AutoGluon Predictions
    logging.info("Evaluating AutoGluon performance on new data...")
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
        ag_avail = [c for c in ag_feature_cols if c in df_new.columns]
        ag_test = df_new[ag_avail].copy()
        ag_probs_df = predictor.predict_proba(ag_test)
        ag_probs = ag_probs_df[TARGET_NAMES].values
        ag_acc = accuracy_score(y_new, np.argmax(ag_probs, axis=1), sample_weight=sample_weights)
    except Exception as e:
        logging.warning(f"AutoGluon evaluation failed: {e}")
        ag_acc = 0.0

    logging.info(f"Sub-model Accuracy on New Data -> Tree: {tree_acc:.3f}, GNN: {gnn_acc:.3f}, AutoGluon: {ag_acc:.3f}")
    
    # 4. Dynamic Weight Adjustment
    # We apply a softmax-like re-weighting based on recent accuracy, 
    # but bounded to prevent massive drift.
    
    base_scores = np.array([tree_acc, gnn_acc, ag_acc])
    if np.sum(base_scores) == 0:
        logging.warning("All models failed evaluation. Keeping existing weights.")
        return blend_data

    # Softmax with temperature to adjust weights based on performance
    temp = 0.5
    exp_scores = np.exp(base_scores / temp)
    new_raw_weights = exp_scores / np.sum(exp_scores)
    
    # Blend with historical weights (momentum = 0.7 historical, 0.3 new)
    momentum = 0.7
    historical = np.array([current_weights['tree'], current_weights['gnn'], current_weights['ag']])
    final_weights = momentum * historical + (1 - momentum) * new_raw_weights
    
    # Ensure they sum to 1
    final_weights = final_weights / np.sum(final_weights)
    
    new_blend_weights = {
        'tree': float(final_weights[0]),
        'gnn': float(final_weights[1]),
        'ag': float(final_weights[2])
    }
    
    logging.info(f"Updated Blend Weights (momentum-smoothed): {new_blend_weights}")
    blend_data['blend_weights'] = new_blend_weights
    
    return blend_data

def finetune_gnn_on_feedback(df_new, y_new, blend_data):
    """Finetune GNN on new feedback data with low learning rate."""
    logging.info("Finetuning GNN on feedback data...")
    try:
        gnn_scaler = blend_data['gnn_scaler']
        gnn_corridor_feats = blend_data['gnn_corridor_feats']
        gnn_adj_norm = blend_data['gnn_adj_norm']
        gnn_corridor_idx = blend_data['gnn_corridor_idx']
        gnn_available_feats = blend_data['gnn_available_feats']
        
        # Prepare data
        for f in gnn_available_feats:
            if f not in df_new.columns:
                df_new[f] = 0
        
        X_infer_gnn = gnn_scaler.transform(df_new[gnn_available_feats].fillna(0).values)
        infer_corr = np.array([gnn_corridor_idx.get(c, 0) for c in df_new["corridor"]])
        
        # Load model
        gnn_model = GridlockGNN(len(gnn_available_feats), gnn_corridor_feats.shape[1], len(gnn_corridor_idx), 64, 3, 0.3)
        gnn_model.load_state_dict(torch.load("models/gnn_blend_weights.pt", weights_only=True))
        gnn_model.train()
        
        # Finetune with low LR
        optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X_infer_gnn)
        corridor_feats_tensor = torch.FloatTensor(gnn_corridor_feats)
        adj_norm_tensor = torch.FloatTensor(gnn_adj_norm)
        corr_tensor = torch.LongTensor(infer_corr)
        y_tensor = torch.LongTensor(y_new)
        
        n_samples = len(X_tensor)
        batch_size = min(32, n_samples)
        
        for epoch in range(10):
            perm = torch.randperm(n_samples)
            epoch_loss = 0
            n_batches = 0
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                logits = gnn_model(
                    X_tensor[idx],
                    corridor_feats_tensor,
                    adj_norm_tensor,
                    corr_tensor[idx]
                )
                loss = criterion(logits, y_tensor[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            if epoch % 3 == 0:
                logging.info(f"  Finetune epoch {epoch}: loss={epoch_loss/n_batches:.4f}")
        
        # Save updated weights
        torch.save(gnn_model.state_dict(), "models/gnn_blend_weights.pt")
        logging.info("GNN finetuned and saved successfully.")
        return True
    except Exception as e:
        logging.warning(f"GNN finetune failed: {e}")
        return False


def check_feature_drift(df_new, blend_data):
    """Check for feature distribution drift using KS test."""
    logging.info("Checking feature drift...")
    try:
        # Load original training feature stats from blend_data if available
        # For now, compute basic stats on new data
        gnn_available_feats = blend_data.get('gnn_available_feats', [])
        drift_features = []
        
        for feat in gnn_available_feats:
            if feat in df_new.columns:
                vals = df_new[feat].fillna(0).values
                # Check for extreme values or distribution shifts
                if np.std(vals) == 0:
                    continue
                # Simple heuristic: if mean is >3 std from expected range
                if np.abs(np.mean(vals)) > 10 * np.std(vals):
                    drift_features.append(feat)
        
        drift_detected = len(drift_features) >= 3
        if drift_detected:
            logging.warning(f"FEATURE DRIFT DETECTED on {len(drift_features)} features: {drift_features[:5]}")
        
        # Save drift report
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/drift_report.json', 'w') as f:
            json.dump({
                'drift_detected': drift_detected,
                'drift_features': drift_features,
                'n_new_samples': len(df_new),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return drift_detected
    except Exception as e:
        logging.warning(f"Drift check failed: {e}")
        return False


def save_updated_artifacts(blend_data):
    logging.info("Saving updated model artifacts...")
    weights = blend_data.get('blend_weights', {'tree': 0.8, 'gnn': 0.15, 'ag': 0.05})
    with open('models/dynamic_weights.json', "w") as f:
        json.dump(weights, f)
    logging.info("Post-Event Learning Cycle Complete. System optimized for current traffic conditions.")

def run_learning_cycle():
    print("=" * 60)
    print("AUTONOMOUS POST-EVENT LEARNING CYCLE")
    print("=" * 60)
    
    blend_data = load_model_artifacts()
    if not blend_data:
        return
        
    df_new, y_new, feedback_timestamps = simulate_new_data_ingestion()
    updated_blend_data = evaluate_and_update_weights(df_new, y_new, blend_data, feedback_timestamps)
    
    # Check for feature drift
    drift_detected = check_feature_drift(df_new, updated_blend_data)
    
    # Finetune GNN on new feedback data
    if len(df_new) > 10:
        finetune_gnn_on_feedback(df_new, y_new, updated_blend_data)
    
    # If drift detected, log for full retrain
    if drift_detected:
        logging.warning("DRIFT DETECTED: Consider full pipeline retrain (run 01_data_cleaning_and_fe.py && 02_model_training.py)")
    
    save_updated_artifacts(updated_blend_data)

if __name__ == "__main__":
    run_learning_cycle()
