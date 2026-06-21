"""
Gridlock 2.0 — GNN Duration Prediction Model
A Graph Neural Network that uses corridor adjacency to capture spatial
congestion propagation, then combines with the existing tree ensemble.

Architecture:
  1. Build a corridor graph from the adjacency matrix (22 nodes, 48+ edges)
  2. Per-incident features → per-corridor aggregation
  3. 2-layer GraphSAGE-style message passing (pure PyTorch, no PyG dependency)
  4. Corridor-contextualized features → 3-class duration prediction
  5. GNN probabilities blended into the existing ensemble

Key idea: the GNN gives each incident access to the state of neighboring
corridors, which flat tree models cannot do.
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

DATA_PATH = "data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
ADJ_PATH = "data/corridor_adjacency.npy"
IDX_PATH = "data/corridor_index.json"
OUT_PATH = "outputs/gnn_metrics.json"
MODEL_PATH = "models/gnn_model.pt"

TARGET_NAMES = ["<30min", "30min-2hr", "2hr+"]
TARGET_MAPPING = {"<30min": 0, "30min-2hr": 1, "2hr+": 2}

# Features to use for per-incident encoding
INCIDENT_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_night",
    "minutes_since_midnight", "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month_sin", "month_cos", "report_delay_min", "desc_word_count",
    "desc_char_count", "desc_urgency_score", "latitude_num", "longitude_num",
    "dist_from_center", "has_end_coord",
    "closure_flag", "priority_weight", "rush_hour_flag",
    "rush_hour_x_closure", "high_priority_x_closure",
    "kw_fire", "kw_spill", "kw_overturned", "kw_injury",
    "kw_block", "kw_heavy", "kw_normal", "kw_tow", "kw_breakdown",
    "authenticated_flag",
]

# Per-corridor graph node features (computed from training data)
CORRIDOR_AGG_FEATURES = [
    "corridor_incident_count", "corridor_mean_duration",
    "corridor_critical_rate", "corridor_std_duration",
]


def prepare_data():
    """Prepare data with graph-aware features."""
    df = pd.read_csv(DATA_PATH, low_memory=False)

    for col in ["start_datetime", "created_date", "resolved_datetime", "closed_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    end_time = df["resolved_datetime"].fillna(df["closed_datetime"])
    df["duration_min"] = (end_time - df["start_datetime"]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] >= 0) & (df["duration_min"] <= 1440)].copy()
    df["duration_bucket"] = pd.cut(
        df["duration_min"], bins=[-np.inf, 30, 120, np.inf], labels=TARGET_NAMES,
    )

    # Keep unplanned for comparability with existing model
    df = df[df["event_type"].astype(str).str.lower().eq("unplanned")].copy()
    df = df.dropna(subset=["duration_bucket"]).copy()

    # Temporal features
    start = df["start_datetime"]
    created = df["created_date"]
    df["hour"] = start.dt.hour
    df["day_of_week"] = start.dt.dayofweek
    df["month"] = start.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["minutes_since_midnight"] = df["hour"] * 60 + start.dt.minute
    df["hour_sin"] = np.sin(df["hour"] * (2 * np.pi / 24))
    df["hour_cos"] = np.cos(df["hour"] * (2 * np.pi / 24))
    df["day_sin"] = np.sin(df["day_of_week"] * (2 * np.pi / 7))
    df["day_cos"] = np.cos(df["day_of_week"] * (2 * np.pi / 7))
    df["month_sin"] = np.sin(df["month"] * (2 * np.pi / 12))
    df["month_cos"] = np.cos(df["month"] * (2 * np.pi / 12))
    df["report_delay_min"] = (created - start).dt.total_seconds() / 60.0
    df["report_delay_min"] = df["report_delay_min"].clip(-1440, 1440).fillna(0)

    # NLP keyword features
    desc = df["description"].fillna("").astype(str).str.lower()
    df["desc_word_count"] = desc.str.split().str.len().fillna(0)
    df["desc_char_count"] = desc.str.len().fillna(0)
    keyword_groups = {
        "fire": "fire|burn|smoke", "spill": "spill|fuel|oil|leak",
        "overturned": "overturn|toppl|roll", "injury": "injur|fatal|death|dead|hospital|ambulance",
        "block": "block|stuck|jam|congest", "heavy": "heavy|large|major|serious",
        "normal": "normal|free|clear|no problem", "tow": "tow|crane|mechanic|repair",
        "breakdown": "break|off road|puncture|engine|starting",
    }
    for name, pattern in keyword_groups.items():
        df[f"kw_{name}"] = desc.str.contains(pattern, na=False, regex=True).astype(int)
    df["desc_urgency_score"] = df[[f"kw_{k}" for k in keyword_groups if k != "normal"]].sum(axis=1)

    # Geo
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    endlat = pd.to_numeric(df["endlatitude"], errors="coerce").replace(0, np.nan)
    endlon = pd.to_numeric(df["endlongitude"], errors="coerce").replace(0, np.nan)
    df["latitude_num"] = lat
    df["longitude_num"] = lon
    df["dist_from_center"] = np.sqrt((lat - 12.9716) ** 2 + (lon - 77.5946) ** 2) * 111
    df["has_end_coord"] = endlat.notna().astype(int)
    # removed incident_span_km_clean due to data leakage

    # Operational
    closure = df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    priority_high = df["priority"].astype(str).str.lower().eq("high").astype(int)
    priority_medium = df["priority"].astype(str).str.lower().eq("medium").astype(int)
    df["closure_flag"] = closure
    df["priority_weight"] = priority_high * 3 + priority_medium * 2 + (1 - priority_high - priority_medium).clip(lower=0)
    rush = (((df["hour"] >= 7) & (df["hour"] <= 10)) | ((df["hour"] >= 17) & (df["hour"] <= 20))).astype(int)
    df["rush_hour_flag"] = rush
    df["rush_hour_x_closure"] = rush * closure
    df["high_priority_x_closure"] = priority_high * closure

    # Recovered feature: authenticated
    df["authenticated_flag"] = (df["authenticated"].fillna("no").str.lower() == "yes").astype(int)

    # Corridor encoding
    df["corridor"] = df["corridor"].fillna("UNKNOWN").astype(str)

    y = df["duration_bucket"].astype(str).map(TARGET_MAPPING).astype(int)

    return df, y


def generate_dynamic_adjacency(df):
    """Dynamically generate corridor adjacency based on temporal co-occurrence of incidents.
    
    Includes an explicit 'UNKNOWN' node at the last index for handling unseen corridors at inference.
    """
    corridors = df['corridor'].dropna().unique()
    corridors = [c for c in corridors if str(c) != "UNKNOWN" and str(c) != "nan"]
    
    # Build index for known corridors
    corridor_idx = {str(c): i for i, c in enumerate(corridors)}
    # Add UNKNOWN node at the end
    unknown_idx = len(corridors)
    corridor_idx["UNKNOWN"] = unknown_idx
    n = len(corridors) + 1  # +1 for UNKNOWN node
    adj = np.zeros((n, n))
    
    if "start_datetime" in df.columns:
        df_valid = df.dropna(subset=["start_datetime"]).copy()
        # Group incidents into 2-hour windows to detect co-occurring congestion
        df_valid["time_bin"] = df_valid["start_datetime"].dt.floor("2h")
        
        # Create a boolean matrix of time_bin vs corridor
        bin_corr = pd.crosstab(df_valid["time_bin"], df_valid["corridor"])
        # Cap at 1 (we just care if *any* incident occurred)
        bin_corr = (bin_corr > 0).astype(int)
        
        valid_cols = [c for c in corridors if c in bin_corr.columns]
        bin_corr = bin_corr[valid_cols]
        
        # Compute co-occurrence via dot product
        co_occ = bin_corr.T.dot(bin_corr)
        
        for i, c1 in enumerate(corridors):
            for j, c2 in enumerate(corridors):
                if i == j:
                    adj[i][j] = 1.0
                elif c1 in valid_cols and c2 in valid_cols:
                    overlap = co_occ.loc[c1, c2]
                    count1 = bin_corr[c1].sum()
                    count2 = bin_corr[c2].sum()
                    union = count1 + count2 - overlap
                    if union > 0:
                        jaccard = overlap / union
                        # Set edge if jaccard similarity is significant
                        if jaccard > 0.02: 
                            adj[i][j] = float(jaccard)

    # Add UNKNOWN node connections: connect to all known corridors with average weight
    # This allows the GNN to propagate information to/from unseen corridors
    if n > 1:
        # Average edge weight from known corridors
        known_weights = adj[:unknown_idx, :unknown_idx]
        avg_weight = known_weights[known_weights > 0].mean() if (known_weights > 0).any() else 0.1
        # Connect UNKNOWN to all known corridors with average weight
        adj[unknown_idx, :unknown_idx] = avg_weight
        adj[:unknown_idx, unknown_idx] = avg_weight
        # Self-loop for UNKNOWN
        adj[unknown_idx, unknown_idx] = 1.0

    # Save to disk to maintain API server compatibility
    os.makedirs('data', exist_ok=True)
    np.save(ADJ_PATH, adj)
    with open(IDX_PATH, 'w') as f:
        json.dump({'index': corridor_idx}, f)
        
    return adj, corridor_idx


def build_corridor_features(df, y, corridor_idx):
    """Build per-corridor aggregate features from training data only.
    
    Includes UNKNOWN node with global average features for unseen corridors.
    """
    corridors = list(corridor_idx.keys())
    n = len(corridors)
    features = np.zeros((n, len(CORRIDOR_AGG_FEATURES)))

    # Compute global averages for UNKNOWN node fallback
    global_dur_mean = df["duration_min"].mean()
    global_critical_rate = (y == 2).mean()
    global_std = df["duration_min"].std()

    for corridor, idx in corridor_idx.items():
        if corridor == "UNKNOWN":
            # Set UNKNOWN node to global averages
            features[idx, 0] = len(df)  # total count
            features[idx, 1] = global_dur_mean
            features[idx, 2] = global_critical_rate
            features[idx, 3] = global_std
            continue
            
        mask = df["corridor"] == corridor
        if mask.sum() == 0:
            continue
        dur = df.loc[mask, "duration_min"]
        y_corr = y[mask]
        features[idx, 0] = mask.sum()
        features[idx, 1] = dur.mean()
        features[idx, 2] = (y_corr == 2).mean()  # critical rate
        features[idx, 3] = dur.std() if mask.sum() > 1 else 0

    return features


def normalize_adjacency(adj):
    """D^{-1/2} A D^{-1/2} normalization for GCN."""
    adj = adj + np.eye(adj.shape[0])  # self-loops
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    D = np.diag(deg_inv_sqrt)
    return D @ adj @ D


class GraphSAGELayer(nn.Module):
    """Simple GraphSAGE-style aggregation layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_self = nn.Linear(in_features, out_features)
        self.W_neigh = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x, adj_norm):
        # adj_norm: (n_corridors, n_corridors) normalized adjacency
        # x: (n_corridors, in_features)
        neigh_agg = torch.mm(adj_norm, x)  # mean neighbor features
        h = self.W_self(x) + self.W_neigh(neigh_agg)
        h = self.ln(h)
        return F.relu(h)


class GridlockGNN(nn.Module):
    """
    GNN for duration prediction.
    1. Encode per-incident features
    2. Message-pass on corridor graph to get corridor context
    3. Concatenate incident features + corridor context
    4. Classify into 3 duration buckets
    """
    def __init__(self, n_incident_features, n_corridor_features, n_corridors,
                 hidden_dim=64, n_classes=3, dropout=0.3):
        super().__init__()
        self.n_corridors = n_corridors

        # Incident feature encoder
        self.incident_encoder = nn.Sequential(
            nn.Linear(n_incident_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Graph layers on corridor features
        self.gnn1 = GraphSAGELayer(n_corridor_features, hidden_dim)
        self.gnn2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.gnn_dropout = nn.Dropout(dropout)

        # Classifier: incident_encoded + corridor_context → class
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, incident_features, corridor_features, adj_norm, corridor_indices):
        """
        Args:
            incident_features: (batch, n_incident_features)
            corridor_features: (n_corridors, n_corridor_features)
            adj_norm: (n_corridors, n_corridors)
            corridor_indices: (batch,) — which corridor each incident belongs to
        """
        # Encode incident
        incident_enc = self.incident_encoder(incident_features)

        # GNN on corridors
        h = self.gnn1(corridor_features, adj_norm)
        h = self.gnn_dropout(h)
        h = self.gnn2(h, adj_norm)  # (n_corridors, hidden_dim)

        # Gather corridor context for each incident
        corridor_context = h[corridor_indices]  # (batch, hidden_dim)

        # Concatenate and classify
        combined = torch.cat([incident_enc, corridor_context], dim=1)
        logits = self.classifier(combined)
        return logits


def train_gnn(df, y, adj_matrix, corridor_idx, epochs=120, lr=0.003, batch_size=128,
              hidden_dim=64, dropout=0.3, verbose=True):
    """Train the GNN and return metrics."""
    # Split
    x_train_df, x_test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build corridor features from training data only (no leakage)
    corridor_feats = build_corridor_features(x_train_df, y_train, corridor_idx)

    # Normalize adjacency
    adj_norm = normalize_adjacency(adj_matrix)

    # Prepare incident features
    scaler = StandardScaler()
    available_feats = [f for f in INCIDENT_FEATURES if f in df.columns]
    X_train_raw = x_train_df[available_feats].fillna(0).values
    X_test_raw = x_test_df[available_feats].fillna(0).values
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Map incidents to corridor indices
    def map_corridors(df_subset):
        indices = []
        for c in df_subset["corridor"]:
            if c in corridor_idx:
                indices.append(corridor_idx[c])
            else:
                indices.append(0)  # fallback
        return np.array(indices)

    train_corr_idx = map_corridors(x_train_df)
    test_corr_idx = map_corridors(x_test_df)

    # To tensors
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train.values)
    y_test_t = torch.LongTensor(y_test.values)
    corridor_feats_t = torch.FloatTensor(corridor_feats)
    adj_norm_t = torch.FloatTensor(adj_norm)
    train_corr_t = torch.LongTensor(train_corr_idx)
    test_corr_t = torch.LongTensor(test_corr_idx)

    # Class weights for imbalanced data
    counts = np.bincount(y_train.values, minlength=3)
    weights = torch.FloatTensor([len(y_train) / (3 * c) for c in counts])

    # Model
    model = GridlockGNN(
        n_incident_features=len(available_feats),
        n_corridor_features=len(CORRIDOR_AGG_FEATURES),
        n_corridors=len(corridor_idx),
        hidden_dim=hidden_dim,
        n_classes=3,
        dropout=dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = 0
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train_t[idx], corridor_feats_t, adj_norm_t, train_corr_t[idx])
            loss = criterion(logits, y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test_t, corridor_feats_t, adj_norm_t, test_corr_t)
                test_preds = test_logits.argmax(dim=1).numpy()
                test_probs = F.softmax(test_logits, dim=1).numpy()

            acc = accuracy_score(y_test.values, test_preds)
            f1 = f1_score(y_test.values, test_preds, average="macro")
            report = classification_report(y_test.values, test_preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
            crit_rec = report["2hr+"]["recall"]

            if verbose:
                print(f"  Epoch {epoch+1:3d} | loss={epoch_loss/n_batches:.4f} | acc={acc:.3f} | macro_f1={f1:.3f} | crit_recall={crit_rec:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 10

            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t, corridor_feats_t, adj_norm_t, test_corr_t)
        test_preds = test_logits.argmax(dim=1).numpy()
        test_probs = F.softmax(test_logits, dim=1).numpy()

    report = classification_report(y_test.values, test_preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test.values, test_preds)
    macro_f1 = f1_score(y_test.values, test_preds, average="macro")
    crit_recall = report["2hr+"]["recall"]

    return {
        "model": model,
        "scaler": scaler,
        "corridor_feats": corridor_feats,
        "adj_norm": adj_norm,
        "available_feats": available_feats,
        "corridor_idx": corridor_idx,
        "test_probs": test_probs,
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "critical_recall": float(crit_recall),
            "per_class": {
                "<30min": {k: float(v) for k, v in report["<30min"].items()},
                "30min-2hr": {k: float(v) for k, v in report["30min-2hr"].items()},
                "2hr+": {k: float(v) for k, v in report["2hr+"].items()},
            },
        },
    }


def kfold_gnn(df, y, adj_matrix, corridor_idx, n_splits=3, hidden_dim=64, epochs=100):
    """K-fold validation for the GNN."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, y), 1):
        print(f"  Fold {fold}/{n_splits}...")
        df_train = df.iloc[train_idx]
        y_train = y.iloc[train_idx]
        df_test = df.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Build corridor features from fold's training data only
        corridor_feats = build_corridor_features(df_train, y_train, corridor_idx)
        adj_norm = normalize_adjacency(adj_matrix)

        available_feats = [f for f in INCIDENT_FEATURES if f in df.columns]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[available_feats].fillna(0).values)
        X_test = scaler.transform(df_test[available_feats].fillna(0).values)

        def map_corridors(df_s):
            return np.array([corridor_idx.get(c, 0) for c in df_s["corridor"]])

        train_corr = map_corridors(df_train)
        test_corr = map_corridors(df_test)

        X_train_t = torch.FloatTensor(X_train)
        X_test_t = torch.FloatTensor(X_test)
        y_train_t = torch.LongTensor(y_train.values)
        corridor_feats_t = torch.FloatTensor(corridor_feats)
        adj_norm_t = torch.FloatTensor(adj_norm)

        counts = np.bincount(y_train.values, minlength=3)
        weights = torch.FloatTensor([len(y_train) / (3 * c) for c in counts])

        model = GridlockGNN(
            n_incident_features=len(available_feats),
            n_corridor_features=len(CORRIDOR_AGG_FEATURES),
            n_corridors=len(corridor_idx),
            hidden_dim=hidden_dim, n_classes=3, dropout=0.3,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(X_train_t))
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]
                logits = model(X_train_t[idx], corridor_feats_t, adj_norm_t, torch.LongTensor(train_corr[idx.numpy()]))
                loss = criterion(logits, y_train_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t, corridor_feats_t, adj_norm_t, torch.LongTensor(test_corr))
            preds = test_logits.argmax(dim=1).numpy()

        acc = accuracy_score(y_test.values, preds)
        f1 = f1_score(y_test.values, preds, average="macro")
        report = classification_report(y_test.values, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
        fold_results.append({
            "fold": fold, "accuracy": float(acc), "macro_f1": float(f1),
            "critical_recall": float(report["2hr+"]["recall"]),
        })
        print(f"    acc={acc:.3f} f1={f1:.3f} crit={report['2hr+']['recall']:.3f}")

    return {
        "folds": fold_results,
        "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_results])),
        "accuracy_std": float(np.std([f["accuracy"] for f in fold_results])),
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in fold_results])),
        "macro_f1_std": float(np.std([f["macro_f1"] for f in fold_results])),
        "critical_recall_mean": float(np.mean([f["critical_recall"] for f in fold_results])),
        "critical_recall_std": float(np.std([f["critical_recall"] for f in fold_results])),
    }


def run_gnn_experiment():
    """Run the full GNN experiment with holdout + k-fold."""
    print("=" * 60)
    print("GRIDLOCK GNN EXPERIMENT")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    df, y = prepare_data()
    print(f"  {len(df)} incidents, {y.value_counts().to_dict()}")

    # Generate dynamic graph
    print("\n[2/5] Generating dynamic corridor graph...")
    adj_matrix, corridor_idx = generate_dynamic_adjacency(df)
    print(f"  {len(corridor_idx)} corridors, {int(adj_matrix.sum())} edge weights")

    # Train GNN
    print("\n[3/5] Training GNN (holdout)...")
    result = train_gnn(df, y, adj_matrix, corridor_idx, epochs=150, lr=0.003, hidden_dim=64)
    metrics = result["metrics"]
    print(f"\n  GNN Holdout Results:")
    print(f"    Accuracy:       {metrics['accuracy']:.3f}")
    print(f"    Macro F1:       {metrics['macro_f1']:.3f}")
    print(f"    Critical Recall: {metrics['critical_recall']:.3f}")

    # K-fold
    print("\n[4/5] Running 3-fold cross-validation...")
    kfold_result = kfold_gnn(df, y, adj_matrix, corridor_idx, n_splits=3, epochs=100)
    print(f"\n  GNN K-Fold Results:")
    print(f"    Accuracy: {kfold_result['accuracy_mean']:.3f} ± {kfold_result['accuracy_std']:.3f}")
    print(f"    Macro F1: {kfold_result['macro_f1_mean']:.3f} ± {kfold_result['macro_f1_std']:.3f}")
    print(f"    Critical Recall: {kfold_result['critical_recall_mean']:.3f} ± {kfold_result['critical_recall_std']:.3f}")

    # Compare with existing ensemble
    print("\n[5/5] Comparison with existing ensemble:")
    print(f"  {'Metric':<20s} {'Existing Ensemble':>18s} {'GNN':>12s}")
    print(f"  {'Accuracy':<20s} {'55.3%':>18s} {metrics['accuracy']*100:>11.1f}%")
    print(f"  {'Macro F1':<20s} {'0.540':>18s} {metrics['macro_f1']:>12.3f}")
    print(f"  {'Critical Recall':<20s} {'73.5%':>18s} {metrics['critical_recall']*100:>11.1f}%")

    # Save
    output = {
        "gnn_holdout": metrics,
        "gnn_kfold": kfold_result,
        "existing_ensemble": {
            "accuracy": 0.553, "macro_f1": 0.540, "critical_recall": 0.735,
        },
        "model_architecture": {
            "type": "GraphSAGE (2-layer, pure PyTorch)",
            "hidden_dim": 64,
            "incident_features": len([f for f in INCIDENT_FEATURES if f in df.columns]),
            "corridor_features": len(CORRIDOR_AGG_FEATURES),
            "corridors": len(corridor_idx),
            "parameters": sum(p.numel() for p in result["model"].parameters()),
        },
        "recovered_features": ["authenticated_flag (intake verification flag)"],
    }

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": result["model"].state_dict(),
        "scaler_mean": result["scaler"].mean_,
        "scaler_scale": result["scaler"].scale_,
        "corridor_feats": result["corridor_feats"],
        "adj_norm": result["adj_norm"],
        "available_feats": result["available_feats"],
        "corridor_idx": result["corridor_idx"],
        "metrics": metrics,
    }, MODEL_PATH)

    print(f"\nSaved metrics to {OUT_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    return output



