# 🛠️ Gridlock 2.0 Setup & Installation Guide

This guide covers the full local deployment of the Gridlock 2.0 architecture, including the ML backend, optimization engine, and the React command center.

## 1. Prerequisites

Before beginning, ensure your system has the following installed:
- **Python**: 3.10 or higher
- **Node.js**: v18.0 or higher
- **Git**: For version control

## 2. Clone the Repository

```bash
git clone https://github.com/nikunjkaushik20/gridlock.git
cd gridlock/round2
```

## 3. Backend Setup (FastAPI & ML Models)

The backend handles the PyTorch GNN, AutoGluon models, and the MILP PuLP Optimizer. 

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

### Start the Backend Server

```bash
# Start the FastAPI Uvicorn Server (Binds to 0.0.0.0 for cross-device access)
python api_server.py
```
> **Note:** The server will output `Loading ML models...` on startup. The GNN and AutoGluon models take a moment to load into memory. Wait for the `Models loaded successfully. API ready at http://localhost:8000` message before making requests.

## 4. Frontend Setup (React + Vite)

Open a **new terminal window** (keep the backend server running).

```bash
cd gridlock/round2/frontend

# Install Node modules
npm install

# Start the Vite Development Server
npm run dev
```
Navigate your browser to `http://localhost:5173` to access the Command Center UI.

## 5. Simulating a Real-Time Event Stream

To see the dashboard react to live events, we provide a mock ingestion script that replays the Astram dataset chronologically via WebSockets.

Open a **third terminal window** (with the python virtual environment activated):

```bash
cd gridlock/round2

# Start the chronological telemetry replay
python mock_realtime_ingest.py
```
*Tip: While this script is running, you can press `b` and hit Enter in the terminal to instantly inject a high-priority "Black Swan" event (chemical spill) and watch the UI and GNN network cascade react.*

## 6. Training from Scratch (Optional)

The repository comes with pre-trained models. However, if you have updated the Astram dataset and want to run the full training pipeline, execute the scripts in this exact order:

```bash
# 1. Clean data and engineer temporal/spatial features
python 01_data_cleaning_and_fe.py

# 2. Train the Tree Ensembles, AutoGluon Stack, and PyTorch GNN
python 02_model_training.py

# 3. Build the endogenous topology graph for routing
python build_endogenous_graph.py

# 4. Trigger the first MLOps calibration cycle
python 04_post_event_learning.py
```
