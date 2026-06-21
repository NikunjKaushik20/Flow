<div align="center">
  <img src="https://via.placeholder.com/1200x300/1a1a2e/47D18C?text=Flow+Command+Center" alt="Flow Banner" />

  <h1>🚦 Flow</h1>
  <p><strong>Autonomous Traffic Incident Triage & Network Optimization Engine</strong></p>

  <p>
    <a href="#-the-problem-the-accuracy-trap"><img src="https://img.shields.io/badge/Status-Production%20Ready-success" alt="Status"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/Architecture-FastAPI%20%7C%20React-blue" alt="Architecture"></a>
    <a href="#layer-2-the-multi-modal-ai-ensemble"><img src="https://img.shields.io/badge/ML-PyTorch%20%7C%20AutoGluon%20%7C%20CatBoost-orange" alt="ML"></a>
    <a href="#layer-3-operations-research--optimization"><img src="https://img.shields.io/badge/Optimization-PuLP%20%28MILP%29-red" alt="Optimization"></a>
  </p>
</div>

<br/>

## 📖 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [The Problem & The Accuracy Trap](#-the-problem--the-accuracy-trap)
3. [Deep Dive: The 5-Layer AI Pipeline](#-deep-dive-the-5-layer-ai-pipeline)
    - [Layer 1: Data Engineering & NLP](#layer-1-data-engineering--nlp)
    - [Layer 2: Multi-Modal Ensemble (80+15+5)](#layer-2-the-multi-modal-ai-ensemble)
    - [Layer 3: MILP Resource Optimizer](#layer-3-operations-research--optimization)
    - [Layer 4: Network Simulation](#layer-4-congestion-simulation)
    - [Layer 5: Continuous MLOps](#layer-5-post-event-learning--mlops)
4. [Performance & Exhaustive ML Search](#-performance--exhaustive-ml-search)
5. [The React Command Center](#-the-react-command-center)
6. [Deployment & Installation](#-deployment--installation)

---

## 🚀 Executive Summary

**Flow** is an end-to-end, AI-driven Command Center designed for modern urban traffic police and city planners. Built strictly within the constraints of the provided Astram dataset, Flow bridges the gap between **incoming urban incident reports** and **autonomous ground-force deployment**. 

The dataset lacks live congestion state, field response time, weather, and actual vehicle density, which imposes a real ceiling on purely historical models. **Flow handles this honestly, avoids data leakage, and converts uncertain forecasts into actionable manpower, barricading, and diversion recommendations.**

It ingests incoming incidents, predicts clearance durations using a state-of-the-art **Multi-Modal AI Ensemble**, mathematically computes spatial spillover, and solves a Mixed-Integer Linear Program (MILP) to dispatch officers exactly where needed.

---

## 🎯 The Problem & The "Accuracy Trap"

**Operational Challenge:** Urban events (rallies, crashes, construction) create chaotic, localized traffic breakdowns. Today, resource deployment is based on manual guesswork, lacks spatial awareness, and has no automated post-event learning loop.

### ⚠️ We found the 100% accuracy trap and chose deployable forecasting instead.

During our audit, we identified a massive **data leakage** flaw inherent in naive hackathon submissions.
* **The Leaky Scorecard:** It is easy to achieve **100% accuracy** by feeding models fields like `resolved_datetime` or `closed_datetime`. However, these fields are known *only after* the incident ends. A model using these looks perfect on paper but fails completely in a live control room.
* **The Deployable Scorecard (Our Approach):** We explicitly stripped all post-resolution fields. Our model relies **only** on pre-event and incident-intake features. 

**Our Philosophy:** *For traffic police, a modest, honest forecast 30 minutes before the event is infinitely more valuable than a perfect answer 2 hours after the event.*

---

## 🧠 Deep Dive: The 5-Layer AI Pipeline

### Layer 1: Data Engineering & NLP
Traffic data is inherently noisy. We process raw dumps of incident reports through a rigorous extraction pipeline:
- **NLP Urgency Extraction**: Uses optimized keyword parsing to detect operational triggers (`kw_overturned`, `kw_rally`, `kw_waterlogging`).
- **Geospatial Anchoring**: Computes Haversine distances to major city hubs.
- **Temporal Harmonics**: Encodes `hour` and `day_of_week` using Sine/Cosine cyclical transformations to preserve the continuous nature of time.

### Layer 2: The Multi-Modal AI Ensemble
Traditional models treat traffic incidents as isolated rows. Flow understands traffic is a **physical, geometric network**. After testing over 230 configurations, we settled on an **80+15+5 blending strategy** to achieve optimal accuracy while maintaining over 70% recall on catastrophic 2hr+ jams:

1. **Optimized Tree Ensembles — 80% Weight**
   - A heavy blend of LightGBM + AdaBoost + CatBoost. Excellent at handling tabular distributions and dense categorical embeddings.
2. **PyTorch GraphSAGE (GNN) — 15% Weight**
   - Solves the **Spatial Isolation Problem**. We built a standalone urban corridor adjacency graph endogenously from the dataset. It mathematically calculates how an incident at a bottleneck chokes adjacent arterial roads. Validated explicitly via rigorous **held-out unseen corridor validation** to prove out-of-distribution generalization. Unknown corridors now gracefully fallback to nearest-neighbor spatial centroids rather than silent indices.
3. **AutoGluon L2-Stacked Meta-Learner — 5% Weight**
   - Runs a NeuralNetFastAI + XGBoost + LightGBM stack to pick up deep, non-linear interactions.

### Layer 3: Operations Research & Optimization
Predicting a 2-hour jam is useless without tactical action. The **Dynamic Resource Optimizer** (`optimization_engine.py`) takes the ML output and acts on it using `PuLP` (MILP solver).
- **Objective**: Minimize Total Incident Response Time & Maximize High-Severity Coverage.
- **Data-Derived Constraints**: Instead of arbitrary heuristics, all rules for Barricades and Manpower are **learned directly from historical data**. We extracted real Police Station coordinates from the dataset to act as deployment hubs for the MILP solver.
- **Output**: Produces explicit, quantified deployment strategies for Manpower (officers) and Barricading.

### Layer 4: Spillover-Aware Endogenous Routing & Simulation
Using Bureau of Public Roads (BPR) delay formulas, the `/api/simulate` endpoint artificially closes lanes on our endogenous network graph and calculates the ripple effect radiating through neighboring corridors. 
Crucially, **diversion routing is 100% data-derived**. The system routes traffic using our endogenous graph penalized by **learned historical spillover weights** (how often two corridors congest within 30 mins of each other), completely avoiding any "outsourced reasoning" via external APIs.

### Layer 5: Post-Event Learning & GNN Finetuning (MLOps)
Hackathon models die in production. Flow survives. 
The `04_post_event_learning.py` loop autonomously ingests newly resolved incidents daily from an SQLite database. It calculates explicit feature drift and automatically adjusts ensemble blending weights via softmax. Further, it triggers **on-the-fly PyTorch finetuning** of the GNN to adapt to shifting city traffic patterns permanently.

---

## 📊 Performance & Exhaustive ML Search

We executed a genuinely exhaustive model search on the 2,503 valid unplanned incidents using strictly leakage-free intake features.

### The Search Results

| Family | Models Tested | Best Result |
|---|---|---|
| **Boosted trees** | CatBoost, LightGBM, XGBoost, AdaBoost | ✅ **Best baseline family** (55-58% acc) |
| **Random forests** | RandomForest, ExtraTrees, HistGradient | 51-55% acc |
| **Classical/Linear** | Logistic Regression, Naive Bayes | 49-52% acc |
| **Deep tabular** | MLP, FT-Transformer | 44-48% acc |
| **Foundation Models**| TabPFN v3 | 54% acc, **1.5% critical recall 💀** |
| **GNN** | 2-layer GraphSAGE on endogenous graph | 50% acc solo, **+2% as blend** |
| **Auto-ML** | AutoGluon best_quality (L2 stacking) | 54% acc |
| **Multi-model Blends**| 230+ combinations tested | **Tree 80% + GNN 15% + AG 5%** |

### Final Selected Model: The 3-Way Blend
On the provided dataset, our leakage-free model reaches **57.7% holdout accuracy**, a **0.549 macro-F1**, and a critical operational metric of **70.6% recall on catastrophic 2hr+ incidents**. 

No individual model family—including the literature's top small-data recommendations like TabPFN—meaningfully exceeded this. Blending the tabular strength of trees with the spatial congestion propagation of the GNN pushed us to the absolute ceiling for this dataset.

---

## 💻 The React Command Center

The frontend (`frontend/`) is a stunning, dark-mode-first React + Vite dashboard designed for operational clarity. 

### Key Modules:
- **Command Center Live Map**: A Leaflet.js interactive map plotting all active network corridors. Corridors glow Green, Yellow, or Red based on computed spatial risk models. Includes an **Endogenous Graph Visualizer** with a toggle to overlay learned historical spillover weights.
- **Incident Drawer**: Clicking "Predict" slides out a tactical drawer containing the AI's confidence intervals, cascade metrics, and explicit operational directives with exact coordinates for barricade deployment and endogenous spillover-aware diversion routing. We added strict transparency badges (**"Data-Derived"** vs **"Heuristic"**) so control room operators know exactly what is AI-driven versus formulaic.
- **Impact Simulator Sandbox**: Allows urban planners to inject a hypothetical crisis (e.g., "Chemical Spill on Ring Road") and watch the predicted congestion radiate through the graph using the MILP optimizer.

---

## ⚙️ Deployment & Installation

Detailed installation instructions are available in [setup.md](setup.md).

Quick start:
```bash
# Start Backend
pip install -r requirements.txt
python api_server.py

# Start Frontend (New Terminal)
cd frontend
npm install
npm run dev

# Start Mock Telemetry Replay (New Terminal)
python mock_realtime_ingest.py
```

---

<div align="center">
  <p>Built for the <b>Flipkart Gridlock 2.0 Hackathon</b>.</p>
  <p>MIT License</p>
</div>
