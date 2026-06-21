import pandas as pd
import numpy as np
import json
import logging
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline():
    # 1. Load Data
    data_path = 'data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
    if not os.path.exists(data_path):
        logging.error(f"Dataset {data_path} not found.")
        return

    logging.info("Loading dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    logging.info(f"Original shape: {df.shape}")

    logging.info("Engineering target variables (duration_min, duration_bucket)...")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['resolved_datetime'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')

    # Fallback to closed_datetime if resolved is not available
    df['end_time'] = df['resolved_datetime'].fillna(df['closed_datetime'])
    
    # Duration in minutes
    df['duration_min'] = (df['end_time'] - df['start_datetime']).dt.total_seconds() / 60.0

    # Clean extreme outliers (cap at 24 hours) and completely drop unresolved incidents
    valid_mask = (df['duration_min'] >= 0) & (df['duration_min'] <= 1440)
    df = df[valid_mask].copy()
    
    # Bucket into operational bands
    df['duration_bucket'] = pd.cut(
        df['duration_min'],
        bins=[-np.inf, 30, 120, np.inf],
        labels=['<30min', '30min-2hr', '2hr+']
    )

    # ============================================================
    # NOTE: corridor_hist_mean/median REMOVED from here.
    # They will be computed AFTER train/test split in 02_model_training.py
    # to prevent target leakage.
    # ============================================================

    # ============================================================
    # NOTE: concurrent_density REMOVED from here.
    # It will be computed AFTER train/test split in 02_model_training.py
    # to prevent target leakage.
    # ============================================================

    # 3. Temporal Features (NO LEAKAGE - safe to compute globally)
    logging.info("Extracting temporal features...")
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    def hour_bucket(h):
        if pd.isna(h): return 'unknown'
        elif 7 <= h <= 10: return 'morning_rush'
        elif 11 <= h <= 16: return 'midday'
        elif 17 <= h <= 20: return 'evening_rush'
        elif 21 <= h <= 23: return 'night'
        else: return 'early_morning'

    df['hour_bucket'] = df['hour'].apply(hour_bucket)
    # One-hot encode hour_bucket
    hour_dummies = pd.get_dummies(df['hour_bucket'], prefix='hour_bucket', dummy_na=False)
    df = pd.concat([df, hour_dummies], axis=1)
    df.drop(columns=['hour_bucket'], inplace=True)
    
    df['month'] = df['start_datetime'].dt.month
    
    # Cyclical Temporal Engineering
    logging.info("Applying cyclical temporal transformations...")
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    df['day_sin'] = np.sin(df['day_of_week'] * (2. * np.pi / 7))
    df['day_cos'] = np.cos(df['day_of_week'] * (2. * np.pi / 7))

    # NEW: Additional temporal features
    logging.info("Adding enhanced temporal features...")
    df['is_month_end'] = df['start_datetime'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['start_datetime'].dt.is_month_start.astype(int)
    df['quarter'] = df['start_datetime'].dt.quarter
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['minutes_since_midnight'] = df['hour'] * 60 + df['start_datetime'].dt.minute
    df['month_sin'] = np.sin(df['month'] * (2. * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2. * np.pi / 12))

    # ============================================================
    # NOTE: TF-IDF REMOVED from here.
    # It will be fit on training data only in 02_model_training.py
    # to prevent target leakage.
    # Keep raw description for later use.
    # ============================================================
    
    # NEW: Hand-crafted NLP keyword features (NO LEAKAGE - binary flags)
    logging.info("Extracting NLP keyword binary features...")
    desc_lower = df['description'].fillna('').str.lower()
    df['desc_word_count'] = desc_lower.str.split().str.len().fillna(0).astype(int)
    df['desc_char_count'] = desc_lower.str.len().fillna(0).astype(int)
    df['has_keyword_fire'] = desc_lower.str.contains('fire|burn', na=False).astype(int)
    df['has_keyword_spill'] = desc_lower.str.contains('spill|fuel|oil|leak', na=False).astype(int)
    df['has_keyword_overturned'] = desc_lower.str.contains('overturn|toppl|roll', na=False).astype(int)
    df['has_keyword_injury'] = desc_lower.str.contains('injur|fatal|death|dead|hospital|ambulance', na=False).astype(int)
    df['has_keyword_block'] = desc_lower.str.contains('block|stuck|jam|congest', na=False).astype(int)
    df['has_keyword_heavy'] = desc_lower.str.contains('heavy|large|major|serious', na=False).astype(int)
    df['desc_urgency_score'] = (
        df['has_keyword_fire'] + df['has_keyword_spill'] + 
        df['has_keyword_overturned'] + df['has_keyword_injury'] +
        df['has_keyword_block'] + df['has_keyword_heavy']
    )

    # 4. Leakage Purge
    logging.info("Purging post-resolution and non-actionable features...")
    leakage_cols = [
        'closed_datetime', 'resolved_datetime', 'modified_datetime', 'status', 
        'end_datetime', 'authenticated', 'closed_by_id', 'resolved_by_id', 
        'resolved_at_address', 'resolved_at_latitude', 'resolved_at_longitude', 
        'gba_identifier', 'comment', 'meta_data', 'kgid', 'route_path', 
        'client_id', 'created_by_id', 'last_modified_by_id', 'assigned_to_police_id', 
        'citizen_accident_id', 'end_time'
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # ============================================================
    # NOTE: DBSCAN REMOVED from here.
    # It will be fit on training data only in 02_model_training.py
    # to prevent target leakage.
    # ============================================================

    # NEW: Geospatial distance features (NO LEAKAGE - computed from raw coords)
    logging.info("Computing geospatial distance features...")
    # Distance from dynamic city center
    city_center_lat = df['latitude'].median()
    city_center_lon = df['longitude'].median()
    
    # Fallback to Bangalore if the dataset has no valid coordinates at all
    if pd.isna(city_center_lat) or pd.isna(city_center_lon):
        city_center_lat, city_center_lon = 12.9716, 77.5946

    df['dist_from_center'] = np.sqrt(
        (df['latitude'].fillna(city_center_lat) - city_center_lat)**2 + 
        (df['longitude'].fillna(city_center_lon) - city_center_lon)**2
    ) * 111  # rough km conversion

    # 5. Categorical Encoding
    logging.info("Applying Categorical Encoding...")
    high_cardinality = ['corridor', 'event_cause', 'zone', 'police_station', 'junction']
    encoders = {}
    
    for col in high_cardinality:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN').astype(str)
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            # Store categories list for UI dropdowns/reference
            encoders[col] = [str(c) for c in le.classes_]

    low_cardinality = ['priority', 'veh_type', 'requires_road_closure', 'event_type']
    for col in low_cardinality:
        if col in df.columns:
            # Drop first to avoid collinearity if needed, but for XGBoost standard dummy is fine
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)

    # NEW: Interaction Features (NO LEAKAGE - computed from existing features)
    logging.info("Engineering interaction features...")
    is_rush = ((df['hour'] >= 7) & (df['hour'] <= 10) | (df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['rush_hour_x_closure'] = is_rush * df.get('requires_road_closure_True', pd.Series(0, index=df.index)).fillna(0).astype(int)
    df['corridor_x_cause'] = df['corridor_encoded'] * df['event_cause_encoded']
    df['weekend_x_hour'] = df['is_weekend'] * df['hour']
    
    priority_weight = df.get('priority_High', pd.Series(0, index=df.index)).fillna(0).astype(int) * 3 + \
                     df.get('priority_Medium', pd.Series(0, index=df.index)).fillna(0).astype(int) * 2 + 1
    df['priority_weight'] = priority_weight
    df['cluster_x_cause'] = df.get('zone_encoded', pd.Series(0, index=df.index)) * df['event_cause_encoded']
    df['rush_hour_flag'] = is_rush

    # 6. Split Planned vs Unplanned
    logging.info("Splitting datasets and saving to Parquet...")
    df_planned = df[df['event_type'] == 'planned'].copy()
    df_unplanned = df[df['event_type'] == 'unplanned'].copy()
    
    # NEW: Compute empirical corridor vulnerability metrics (Data-backed proxy for traffic volume)
    logging.info("Computing empirical corridor vulnerability metrics...")
    date_min = df['start_datetime'].min()
    date_max = df['start_datetime'].max()
    days = max(1, (date_max - date_min).days) if pd.notna(date_min) and pd.notna(date_max) else 30
    corridor_counts = df['corridor'].fillna('UNKNOWN').value_counts()
    corridor_vulnerability = (corridor_counts / days).to_dict()
    
    # NEW: Extract REAL police station locations as deployment units (from dataset)
    logging.info("Extracting real police station deployment units from dataset...")
    station_cols = ['police_station', 'latitude', 'longitude']
    station_data = df[station_cols].dropna(subset=['police_station', 'latitude', 'longitude'])
    station_data = station_data[station_data['police_station'] != 'UNKNOWN']
    station_data = station_data[station_data['police_station'] != 'No Police Station']
    police_units = station_data.groupby('police_station').agg({
        'latitude': 'median',
        'longitude': 'median',
        'police_station': 'count'
    }).rename(columns={'police_station': 'incident_count'}).reset_index()
    police_units = police_units[police_units['incident_count'] >= 10]  # Only stations with sufficient coverage
    police_units_list = police_units.to_dict('records')
    logging.info(f"Extracted {len(police_units_list)} real police station deployment units")
    
    # NEW: Learn manpower requirements from historical patterns (duration ~ severity)
    # Since we don't have actual officer counts, we model required officers as function of 
    # predicted duration bucket, event_cause, and corridor historical severity
    logging.info("Learning manpower requirements from historical duration patterns...")
    valid_duration = df.dropna(subset=['duration_min', 'duration_bucket'])
    manpower_lookup = valid_duration.groupby(['event_cause', 'duration_bucket', 'corridor']).agg(
        median_duration=('duration_min', 'median'),
        incident_count=('duration_min', 'count')
    ).reset_index()
    # Heuristic: map duration buckets to officer requirements (learned from data patterns)
    # <30min -> 1-2 officers, 30min-2hr -> 2-4 officers, 2hr+ -> 4-6 officers
    # Adjusted by event_cause historical median duration
    manpower_rules = {}
    for _, row in manpower_lookup.iterrows():
        if row['incident_count'] >= 3:  # Minimum samples
            key = (row['event_cause'], row['duration_bucket'], row['corridor'])
            median_dur = row['median_duration']
            if median_dur <= 30:
                officers = 2
            elif median_dur <= 120:
                officers = 3
            else:
                officers = 5
            manpower_rules[key] = officers
    
    # Global fallback by duration bucket
    bucket_officers = valid_duration.groupby('duration_bucket')['duration_min'].median().to_dict()
    global_fallback = {}
    for bucket, med_dur in bucket_officers.items():
        if med_dur <= 30:
            global_fallback[bucket] = 2
        elif med_dur <= 120:
            global_fallback[bucket] = 3
        else:
            global_fallback[bucket] = 5
    
    # NEW: Learn barricade requirements from requires_road_closure patterns
    logging.info("Learning barricade requirements from road closure patterns...")
    df['requires_closure_bool'] = df['requires_road_closure'].astype(str).str.upper().eq('TRUE')
    barricade_lookup = df.groupby(['event_cause', 'corridor']).agg(
        closure_rate=('requires_closure_bool', 'mean'),
        incident_count=('requires_closure_bool', 'count'),
        median_duration=('duration_min', 'median')
    ).reset_index()
    barricade_rules = {}
    for _, row in barricade_lookup.iterrows():
        if row['incident_count'] >= 3:
            key = (row['event_cause'], row['corridor'])
            # Barricade units based on closure rate and duration
            base_units = int(row['closure_rate'] * 20)  # 0-20 units
            if row['median_duration'] > 120:
                base_units = int(base_units * 1.5)
            barricade_rules[key] = max(0, min(25, base_units))
    
    # Global barricade fallback by event_cause
    global_barricade = df.groupby('event_cause')['requires_closure_bool'].mean().to_dict()
    global_barricade = {k: int(v * 20) for k, v in global_barricade.items()}
    
    # NEW: Compute endogenous graph edge weights from historical spillover (for diversion)
    logging.info("Computing historical spillover weights for diversion routing...")
    # For each incident, check which other corridors had incidents within 30min before/after
    df_sorted = df.sort_values('start_datetime').reset_index(drop=True)
    df_sorted['time_bin'] = df_sorted['start_datetime'].dt.floor('30min')
    
    # Build co-occurrence matrix
    spillover_counts = {}
    for _, group in df_sorted.groupby('time_bin'):
        corridors_in_bin = group['corridor'].unique()
        for c1 in corridors_in_bin:
            for c2 in corridors_in_bin:
                if c1 != c2:
                    key = tuple(sorted([c1, c2]))
                    spillover_counts[key] = spillover_counts.get(key, 0) + 1
    
    # Normalize spillover weights
    max_spillover = max(spillover_counts.values()) if spillover_counts else 1
    spillover_weights = {k: v / max_spillover for k, v in spillover_counts.items()}
    
    # NEW: Compute corridor centroids for nearest-neighbor fallback on unknown corridors
    logging.info("Computing corridor centroids for spatial fallback...")
    corridor_coords = df.groupby('corridor').agg(
        lat=('latitude', 'median'),
        lon=('longitude', 'median'),
        count=('latitude', 'count')
    ).reset_index()
    corridor_coords = corridor_coords[corridor_coords['count'] >= 5]  # Minimum incidents
    corridor_centroids = corridor_coords.set_index('corridor')[['lat', 'lon']].to_dict('index')
    
    # NEW: Compute historical hourly incident baselines for density calculation
    logging.info("Computing historical hourly incident baselines...")
    df['hour'] = df['start_datetime'].dt.hour
    hourly_counts = df.groupby('hour').size()
    date_min = df['start_datetime'].min()
    date_max = df['start_datetime'].max()
    total_days = max(1, (date_max - date_min).days) if pd.notna(date_min) and pd.notna(date_max) else 30
    hourly_baselines = (hourly_counts / total_days).to_dict()
    hourly_baselines = {str(k): float(v) for k, v in hourly_baselines.items()}
    
    # Save Feature Metadata
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/feature_metadata.json', 'w') as f:
        json.dump({
            'high_cardinality_mappings': encoders,
            'city_center': {'lat': float(city_center_lat), 'lon': float(city_center_lon)},
            'corridor_vulnerability': corridor_vulnerability,
            'police_units': police_units_list,
            'manpower_rules': {str(k): v for k, v in manpower_rules.items()},
            'global_manpower_fallback': global_fallback,
            'barricade_rules': {str(k): v for k, v in barricade_rules.items()},
            'global_barricade_fallback': global_barricade,
            'spillover_weights': {f"{k[0]}|{k[1]}": v for k, v in spillover_weights.items()},
            'corridor_centroids': corridor_centroids,
            'hourly_incident_baselines': hourly_baselines
        }, f, indent=4)
        
    try:
        df_unplanned.to_parquet('data/clean_unplanned.parquet', index=False)
        df_planned.to_parquet('data/clean_planned.parquet', index=False)
        logging.info("Successfully saved clean_planned.parquet and clean_unplanned.parquet.")
    except Exception as e:
        logging.error(f"Failed to save parquet files: {e}. Attempting CSV fallback...")
        df_planned.to_csv('clean_planned.csv', index=False)
        df_unplanned.to_csv('clean_unplanned.csv', index=False)
        
    logging.info(f"Pipeline complete. Unplanned events: {len(df_unplanned)}. Planned events: {len(df_planned)}")

if __name__ == "__main__":
    run_pipeline()
