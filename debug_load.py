import pickle
import torch
import importlib
import __main__
print("Starting debug load...")
try:
    mod02 = importlib.import_module("02_model_training")
    setattr(__main__, 'text_col_selector', mod02.text_col_selector)
    setattr(__main__, 'TargetStatsEncoder', mod02.TargetStatsEncoder)
    setattr(__main__, 'to_dense_matrix', mod02.to_dense_matrix)
except Exception as e:
    print(e)
    
from utils_gnn import GridlockGNN
from autogluon.tabular import TabularPredictor

print("Modules imported. Loading pickle...")
with open('models/layer2_blend.pkl', 'rb') as f:
    blend_data = pickle.load(f)
print("Pickle loaded. Initializing GNN...")

gnn_model = GridlockGNN(
    len(blend_data['gnn_available_feats']),
    blend_data['gnn_corridor_feats'].shape[1],
    len(blend_data['gnn_corridor_idx']),
    64, 3, 0.3
)
gnn_model.load_state_dict(torch.load("models/gnn_blend_weights.pt", weights_only=True))
gnn_model.eval()
print("GNN loaded. Initializing AutoGluon...")

print("Calling TabularPredictor.load...")
ag_model = TabularPredictor.load("models/autogluon_gridlock", verbosity=0)
print("AutoGluon loaded. Done.")
