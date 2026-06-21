import pickle
import json
with open('models/layer2_blend.pkl', 'rb') as f:
    blend_data = pickle.load(f)
print("GNN keys:", list(blend_data.get('gnn_corridor_idx', {}).keys())[:5])

with open('outputs/feature_metadata.json', 'r') as f:
    meta = json.load(f)
print("Vuln keys:", list(meta.get('corridor_vulnerability', {}).keys())[:5])

with open('outputs/graph_structure.json', 'r') as f:
    graph = json.load(f)
print("Graph nodes:", [n['id'] for n in graph.get('nodes', [])][:5])
