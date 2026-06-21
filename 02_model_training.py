import os
import gc
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

TARGET_NAMES = ["<30min", "30min-2hr", "2hr+"]
TARGET_MAPPING = {"<30min": 0, "30min-2hr": 1, "2hr+": 2}
DATA_PATH = "data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

from utils_gnn import (
    prepare_data, build_corridor_features, normalize_adjacency, generate_dynamic_adjacency,
    GridlockGNN, INCIDENT_FEATURES, CORRIDOR_AGG_FEATURES
)

class TargetStatsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=25.0):
        self.columns = columns or []
        self.smoothing = smoothing

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy()
        y_arr = np.asarray(y)
        self.global_probs_ = np.bincount(y_arr, minlength=3) / len(y_arr)
        self.maps_ = {}
        for col in self.columns:
            values = frame[col].fillna("MISSING").astype(str)
            mapping = {}
            for value in values.unique():
                mask = values == value
                counts = np.bincount(y_arr[mask], minlength=3)
                probs = (counts + self.smoothing * self.global_probs_) / (counts.sum() + self.smoothing)
                mapping[value] = probs
            self.maps_[col] = mapping
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        parts = []
        for col in self.columns:
            values = frame[col].fillna("MISSING").astype(str)
            parts.append(np.vstack([self.maps_[col].get(v, self.global_probs_) for v in values]))
        return np.hstack(parts) if parts else np.empty((len(frame), 0))


def prepare_frame(unplanned_only=False):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    for col in ["start_datetime", "created_date", "resolved_datetime", "closed_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    end_time = df["resolved_datetime"].fillna(df["closed_datetime"])
    df["duration_min"] = (end_time - df["start_datetime"]).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] >= 0) & (df["duration_min"] <= 1440)].copy()
    df["duration_bucket"] = pd.cut(
        df["duration_min"],
        bins=[-np.inf, 30, 120, np.inf],
        labels=TARGET_NAMES,
    )

    if unplanned_only:
        df = df[df["event_type"].astype(str).str.lower().eq("unplanned")].copy()
    df = df.dropna(subset=["duration_bucket"]).copy()

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
    df["created_hour"] = created.dt.hour
    df["created_day_of_week"] = created.dt.dayofweek
    df["report_delay_min"] = (created - start).dt.total_seconds() / 60.0
    df["report_delay_min"] = df["report_delay_min"].clip(lower=-1440, upper=1440)

    desc = df["description"].fillna("").astype(str).str.lower()
    df["desc_word_count"] = desc.str.split().str.len().fillna(0)
    df["desc_char_count"] = desc.str.len().fillna(0)
    keyword_groups = {
        "fire": "fire|burn|smoke",
        "spill": "spill|fuel|oil|leak",
        "overturned": "overturn|toppl|roll",
        "injury": "injur|fatal|death|dead|hospital|ambulance",
        "block": "block|stuck|jam|congest",
        "heavy": "heavy|large|major|serious",
        "normal": "normal|free|clear|no problem",
        "tow": "tow|crane|mechanic|repair",
        "breakdown": "break|off road|puncture|engine|starting",
    }
    for name, pattern in keyword_groups.items():
        df[f"kw_{name}"] = desc.str.contains(pattern, na=False, regex=True).astype(int)
    df["desc_urgency_score"] = df[[f"kw_{k}" for k in keyword_groups if k != "normal"]].sum(axis=1)

    # Remove leaky incident_span_km computation
    # Treat zero end coordinates as missing. For unplanned incidents these are often placeholders.
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    endlat = pd.to_numeric(df["endlatitude"], errors="coerce").replace(0, np.nan)
    endlon = pd.to_numeric(df["endlongitude"], errors="coerce").replace(0, np.nan)
    df["latitude_num"] = lat
    df["longitude_num"] = lon
    df["dist_from_center"] = np.sqrt((lat - 12.9716) ** 2 + (lon - 77.5946) ** 2) * 111
    df["has_end_coord"] = endlat.notna().astype(int)

    closure = df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    priority_high = df["priority"].astype(str).str.lower().eq("high").astype(int)
    priority_medium = df["priority"].astype(str).str.lower().eq("medium").astype(int)
    df["closure_flag"] = closure
    df["priority_weight"] = priority_high * 3 + priority_medium * 2 + (1 - priority_high - priority_medium).clip(lower=0)
    rush = (((df["hour"] >= 7) & (df["hour"] <= 10)) | ((df["hour"] >= 17) & (df["hour"] <= 20))).astype(int)
    df["rush_hour_flag"] = rush
    df["rush_hour_x_closure"] = rush * closure
    df["high_priority_x_closure"] = priority_high * closure
    df["corridor_cause"] = df["corridor"].fillna("MISSING").astype(str) + "|" + df["event_cause"].fillna("MISSING").astype(str)
    df["police_junction"] = df["police_station"].fillna("MISSING").astype(str) + "|" + df["junction"].fillna("MISSING").astype(str)
    df["address_text"] = df["address"].fillna("").astype(str) + " " + df["end_address"].fillna("").astype(str)

    y = df["duration_bucket"].astype(str).map(TARGET_MAPPING).astype(int)
    return df, y


def text_col_selector(x):
    if isinstance(x, pd.DataFrame):
        series = x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        series = x
    else:
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr[:, 0]
        series = pd.Series(arr)
    return series.fillna("").astype(str)


def to_dense_matrix(x):
    return x.toarray() if sparse.issparse(x) else x


def build_preprocessor(feature_set):
    base_numeric = [
        "latitude_num", "longitude_num", "hour", "day_of_week", "month", "is_weekend", "is_night",
        "minutes_since_midnight", "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "created_hour", "created_day_of_week", "report_delay_min", "desc_word_count", "desc_char_count",
        "dist_from_center", "has_end_coord", "closure_flag", "priority_weight",
        "rush_hour_flag", "rush_hour_x_closure", "high_priority_x_closure", "desc_urgency_score",
    ] + [f"kw_{k}" for k in ["fire", "spill", "overturned", "injury", "block", "heavy", "normal", "tow", "breakdown"]]

    base_cat = [
        "event_cause", "requires_road_closure", "veh_type", "corridor", "priority",
        "police_station", "zone", "junction",
    ]
    extra_cat = ["cargo_material", "reason_breakdown", "age_of_truck"]

    if feature_set == "base":
        cat = base_cat
        text_max = 30
    elif feature_set == "expanded_safe":
        cat = base_cat + extra_cat
        text_max = 80
    else:
        cat = base_cat + extra_cat
        text_max = 150

    target_stat_cols = []
    address_max = 0
    vehicle_cat = []
    if feature_set in ["target_stats", "identity_safe"]:
        target_stat_cols = ["event_cause", "corridor", "police_station", "zone", "junction", "corridor_cause", "police_junction"]
        address_max = 120
    if feature_set == "identity_safe":
        vehicle_cat = ["veh_no"]
        target_stat_cols = target_stat_cols + ["veh_type", "veh_no"]
    if feature_set == "source_ids":
        # Borderline: source/operator IDs may be known at intake, but they are not
        # semantically stable product features and should be disclosed separately.
        cat = cat + ["client_id", "created_by_id", "kgid"]
        target_stat_cols = ["client_id", "created_by_id", "kgid", "event_cause", "corridor", "police_station", "junction"]
        address_max = 120

    transformers = [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), base_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat + vehicle_cat),
        ("word_text", Pipeline([
            ("select", FunctionTransformer(text_col_selector, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=text_max, ngram_range=(1, 2), min_df=3)),
        ]), ["description"]),
        ("char_text", Pipeline([
            ("select", FunctionTransformer(text_col_selector, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=text_max, analyzer="char_wb", ngram_range=(3, 5), min_df=3)),
        ]), ["description"]),
    ]

    if address_max:
        transformers.append(("address_text", Pipeline([
            ("select", FunctionTransformer(text_col_selector, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=address_max, analyzer="char_wb", ngram_range=(3, 5), min_df=3)),
        ]), ["address_text"]))
    if target_stat_cols:
        transformers.append(("target_stats", TargetStatsEncoder(columns=target_stat_cols, smoothing=25), target_stat_cols))

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.8,
    )



NUMERIC_FEATURES = [
    "latitude_num", "longitude_num", "hour", "day_of_week", "month", "is_weekend", "is_night",
    "minutes_since_midnight", "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "created_hour", "created_day_of_week", "report_delay_min", "desc_word_count", "desc_char_count",
    "dist_from_center", "has_end_coord", "closure_flag", "priority_weight",
    "rush_hour_flag", "rush_hour_x_closure", "high_priority_x_closure", "desc_urgency_score",
] + [f"kw_{k}" for k in ["fire", "spill", "overturned", "injury", "block", "heavy", "normal", "tow", "breakdown"]]

CAT_FEATURES = [
    "event_cause", "requires_road_closure", "veh_type", "corridor", "priority",
    "police_station", "zone", "junction", "cargo_material", "reason_breakdown",
    "age_of_truck", "corridor_cause", "police_junction",
]

TEXT_FEATURES = ["description", "address_text"]
CATBOOST_FEATURES = NUMERIC_FEATURES + CAT_FEATURES + TEXT_FEATURES



def prepare_catboost_matrix(df, reference_medians=None):
    x = df[CATBOOST_FEATURES].copy()
    medians = {} if reference_medians is None else dict(reference_medians)
    for col in CAT_FEATURES + TEXT_FEATURES:
        x[col] = x[col].fillna("MISSING").astype(str)
    for col in NUMERIC_FEATURES:
        x[col] = x[col].replace([np.inf, -np.inf], np.nan)
        if col not in medians:
            medians[col] = float(x[col].median()) if x[col].notna().any() else 0.0
        x[col] = x[col].fillna(medians[col])
    return x, medians



def balanced_weights(y):
    counts = np.bincount(y, minlength=3)
    return [len(y) / (3 * c) for c in counts]


def row_weights(y):
    weights = np.asarray(balanced_weights(y))
    return weights[np.asarray(y)]


def make_rf():
    return Pipeline([
        ("prep", build_preprocessor("base")),
        ("model", RandomForestClassifier(
            n_estimators=360, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
    ])


def make_catboost(kind, y, iterations=260):
    weights = {
        "catboost_unweighted": None,
        "catboost_light_critical": [1.0, 1.0, 1.7],
        "catboost_balanced": balanced_weights(y),
    }[kind]
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=6,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=weights,
    )


def train_models(x_train, y_train, cat_iters=260):
    entries = []
    rf = make_rf()
    rf.fit(x_train, y_train)
    entries.append({"name": "rf_base_balanced", "kind": "sklearn", "model": rf})

    sklearn_specs = [
        ("lgbm_text_weighted", "text_rich", LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=260, learning_rate=0.04,
            max_depth=4, num_leaves=18, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, random_state=42, verbose=-1,
        ), True, False),
        ("lgbm_text_unweighted", "text_rich", LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=220, learning_rate=0.04,
            max_depth=4, num_leaves=18, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, random_state=42, verbose=-1,
        ), False, False),
        ("gb_base_weighted", "base", GradientBoostingClassifier(
            n_estimators=140, learning_rate=0.045, max_depth=3,
            subsample=0.85, random_state=42,
        ), True, True),
        ("adaboost_base_weighted", "base", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=8, random_state=42),
            n_estimators=150, learning_rate=0.05, random_state=42,
        ), True, True),
    ]
    for name, feature_set, estimator, use_weights, dense in sklearn_specs:
        steps = [("prep", build_preprocessor(feature_set))]
        if dense:
            steps.append(("dense", FunctionTransformer(to_dense_matrix, accept_sparse=True)))
        steps.append(("model", estimator))
        pipe = Pipeline(steps)
        if use_weights:
            pipe.fit(x_train, y_train, model__sample_weight=row_weights(y_train))
        else:
            pipe.fit(x_train, y_train)
        entries.append({"name": name, "kind": "sklearn", "model": pipe})

    cat_x, medians = prepare_catboost_matrix(x_train)
    cat_idx = [CATBOOST_FEATURES.index(c) for c in CAT_FEATURES]
    text_idx = [CATBOOST_FEATURES.index(c) for c in TEXT_FEATURES]
    for kind in ["catboost_unweighted", "catboost_light_critical", "catboost_balanced"]:
        model = make_catboost(kind, y_train, iterations=cat_iters)
        model.fit(Pool(cat_x, y_train, cat_features=cat_idx, text_features=text_idx))
        entries.append({"name": kind, "kind": "catboost", "model": model, "medians": medians})
    return entries




def proba(entry, x):
    if entry["kind"] == "sklearn":
        return entry["model"].predict_proba(x)
    cat_x, _ = prepare_catboost_matrix(x, reference_medians=entry["medians"])
    cat_idx = [CATBOOST_FEATURES.index(c) for c in CAT_FEATURES]
    text_idx = [CATBOOST_FEATURES.index(c) for c in TEXT_FEATURES]
    return entry["model"].predict_proba(Pool(cat_x, cat_features=cat_idx, text_features=text_idx))

def predict_from_blend(p, ct=None, mt=None):
    preds = np.argmax(p, axis=1)
    if ct is not None:
        preds = preds.copy()
        preds[p[:, 2] >= ct] = 2
    if mt is not None:
        preds = preds.copy()
        non_crit = preds != 2
        preds[non_crit & (p[:, 0] >= mt)] = 0
        preds[non_crit & (p[:, 0] < mt)] = 1
    return preds


def run_training():
    print("=" * 60)
    print("TRAINING 80+15+5 BLEND MODEL")
    print("=" * 60)

    # 1. Prepare Data
    print("\n[1/4] Preparing data...")
    df_tree, y_tree = prepare_frame()
    df_gnn, y_gnn = prepare_data()
    common = df_tree.index.intersection(df_gnn.index)
    df_tree = df_tree.loc[common]
    y_tree = y_tree.loc[common]
    df_gnn = df_gnn.loc[common]
    y_gnn = y_gnn.loc[common]

    x_train_tree, x_test_tree, y_train, y_test = train_test_split(
        df_tree, y_tree, test_size=0.2, random_state=42, stratify=y_tree
    )
    train_idx = x_train_tree.index
    test_idx = x_test_tree.index
    x_train_gnn = df_gnn.loc[train_idx]
    x_test_gnn = df_gnn.loc[test_idx]
    
    # 2. Train Tree Models
    print("\n[2/4] Training Tree Ensemble...")
    entries = train_models(x_train_tree, y_train, cat_iters=300)
    tree_probas = {e["name"]: proba(e, x_test_tree) for e in entries}
    
    best_names = ["lgbm_text_weighted", "adaboost_base_weighted", "catboost_unweighted", "catboost_balanced"]
    available_best = [n for n in best_names if n in tree_probas]
    if not available_best:
        available_best = list(tree_probas.keys())[:4]
    
    w = 1.0 / len(available_best)
    tree_probs = sum(w * tree_probas[n] for n in available_best)
    print(f"  Tree Ensemble Acc: {accuracy_score(y_test, tree_probs.argmax(axis=1)):.3f}")

    # 3. Train GNN Model
    print("\n[3/4] Training GNN Model...")
    adj_matrix, corridor_idx = generate_dynamic_adjacency(x_train_gnn)

    corridor_feats = build_corridor_features(x_train_gnn, y_train, corridor_idx)
    adj_norm = normalize_adjacency(adj_matrix)
    available_feats = [f for f in INCIDENT_FEATURES if f in df_gnn.columns]

    scaler = StandardScaler()
    X_train_gnn = scaler.fit_transform(x_train_gnn[available_feats].fillna(0).values)
    X_test_gnn = scaler.transform(x_test_gnn[available_feats].fillna(0).values)
    train_corr = np.array([corridor_idx.get(c, 0) for c in x_train_gnn["corridor"]])
    test_corr = np.array([corridor_idx.get(c, 0) for c in x_test_gnn["corridor"]])

    counts = np.bincount(y_train.values, minlength=3)
    class_weights = torch.FloatTensor([len(y_train) / (3 * c) for c in counts])

    model = GridlockGNN(len(available_feats), len(CORRIDOR_AGG_FEATURES), len(corridor_idx), 64, 3, 0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for ep in range(120):
        perm = torch.randperm(len(X_train_gnn))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            lo = model(
                torch.FloatTensor(X_train_gnn[idx.numpy()]),
                torch.FloatTensor(corridor_feats),
                torch.FloatTensor(adj_norm),
                torch.LongTensor(train_corr[idx.numpy()])
            )
            loss = criterion(lo, torch.LongTensor(y_train.values[idx.numpy()]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        lo = model(
            torch.FloatTensor(X_test_gnn),
            torch.FloatTensor(corridor_feats),
            torch.FloatTensor(adj_norm),
            torch.LongTensor(test_corr)
        )
        gnn_probs = F.softmax(lo, dim=1).numpy()
    print(f"  GNN Acc: {accuracy_score(y_test, gnn_probs.argmax(1)):.3f}")

    # 4. Load AutoGluon and Blend
    print("\n[4/4] Loading AutoGluon and Creating Final Blend...")
    ag_probs = np.zeros_like(tree_probs)
    try:
        from autogluon.tabular import TabularPredictor
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
        ag_avail = [c for c in ag_feature_cols if c in x_test_tree.columns]
        
        ag_probs_df = predictor.predict_proba(x_test_tree[ag_avail])
        ag_probs = ag_probs_df[TARGET_NAMES].values
        print(f"  AutoGluon Acc: {accuracy_score(y_test, ag_probs.argmax(1)):.3f}")
    except Exception as e:
        print(f"  AutoGluon failed: {e}")

    # 80+15+5 Blend
    blended = 0.80 * tree_probs + 0.15 * gnn_probs + 0.05 * ag_probs
    
    # We use ct=0.26 from multiblend_experiment.py
    ct = 0.26
    mt = None
    
    final_preds = predict_from_blend(blended, ct=ct, mt=mt)
    print("\n=== Final 80+15+5 Blend Performance ===")
    print(classification_report(y_test, final_preds, target_names=TARGET_NAMES))

    # === HELD-OUT CORRIDOR VALIDATION (Generalization Test) ===
    print("\n=== Held-Out Corridor Validation ===")
    # Split by corridor: train on 80% of corridors, test on 20% unseen
    unique_corridors = df_tree['corridor'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_corridors)
    split_idx = int(0.8 * len(unique_corridors))
    train_corridors = set(unique_corridors[:split_idx])
    test_corridors = set(unique_corridors[split_idx:])
    
    corridor_train_mask = df_tree['corridor'].isin(train_corridors)
    corridor_test_mask = df_tree['corridor'].isin(test_corridors)
    
    if corridor_test_mask.sum() > 50:  # Need sufficient test samples
        x_train_corr = df_tree[corridor_train_mask]
        y_train_corr = y_tree[corridor_train_mask]
        x_test_corr = df_tree[corridor_test_mask]
        y_test_corr = y_tree[corridor_test_mask]
        
        print(f"  Train corridors: {len(train_corridors)}, Test corridors: {len(test_corridors)}")
        print(f"  Train samples: {len(x_train_corr)}, Test samples: {len(x_test_corr)}")
        
        # Quick retrain on corridor split for tree models (use best available)
        try:
            # Use a simple RF for speed
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Quick feature prep for corridor split
            feat_cols = [c for c in x_train_corr.columns if c not in ['corridor', 'duration_bucket', 'duration_min', 'start_datetime', 'created_date', 'resolved_datetime', 'closed_datetime', 'end_time']]
            X_tr = x_train_corr[feat_cols].fillna(0)
            X_te = x_test_corr[feat_cols].fillna(0)
            
            # Encode categorical
            for col in X_tr.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_tr[col] = le.fit_transform(X_tr[col].astype(str))
                X_te[col] = le.transform(X_te[col].astype(str))
            
            rf_corr = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            rf_corr.fit(X_tr, y_train_corr)
            corr_preds = rf_corr.predict(X_te)
            corr_acc = accuracy_score(y_test_corr, corr_preds)
            corr_f1 = f1_score(y_test_corr, corr_preds, average='macro')
            
            print(f"  Held-Out Corridor Accuracy: {corr_acc:.3f}")
            print(f"  Held-Out Corridor Macro F1: {corr_f1:.3f}")
            
            # Per-corridor breakdown
            test_df = x_test_corr.copy()
            test_df['pred'] = corr_preds
            test_df['actual'] = y_test_corr.values
            corridor_breakdown = test_df.groupby('corridor').apply(
                lambda g: pd.Series({
                    'acc': accuracy_score(g['actual'], g['pred']),
                    'count': len(g)
                })
            ).sort_values('count', ascending=False)
            print("  Per-corridor breakdown (top 10):")
            print(corridor_breakdown.head(10).to_string())
            
            # Save corridor CV metrics
            import json
            os.makedirs('outputs', exist_ok=True)
            with open('outputs/corridor_cv_metrics.json', 'w') as f:
                json.dump({
                    'held_out_accuracy': float(corr_acc),
                    'held_out_macro_f1': float(corr_f1),
                    'test_corridors': list(test_corridors),
                    'per_corridor': corridor_breakdown.to_dict('index')
                }, f, indent=2)
        except Exception as e:
            print(f"  Corridor CV failed: {e}")
    else:
        print("  Insufficient test samples for corridor split")

    # Save everything
    print("Saving blend models to models/layer2_blend.pkl...")
    
    # Save GNN weights
    torch.save(model.state_dict(), "models/gnn_blend_weights.pt")
    
    # Save all tree models
    selected_entries = [e for e in entries if e["name"] in available_best]
    
    artifact = {
        "tree_entries": selected_entries,
        "tree_names": available_best,
        "gnn_scaler": scaler,
        "gnn_corridor_feats": corridor_feats,
        "gnn_available_feats": available_feats,
        "gnn_adj_norm": adj_norm,
        "gnn_corridor_idx": corridor_idx,
        "blend_weights": {"tree": 0.80, "gnn": 0.15, "ag": 0.05},
        "thresholds": {"ct": ct, "mt": mt},
        "target_names": TARGET_NAMES
    }
    with open("models/layer2_blend.pkl", "wb") as f:
        pickle.dump(artifact, f)
    
    print("Pipeline complete.")

if __name__ == '__main__':
    run_training()
