import json
import networkx as nx
import matplotlib.pyplot as plt
import os

def main():
    print("=== Endogenous Graph Visualizer ===")
    
    graph_path = "outputs/graph_structure.json"
    meta_path = "outputs/feature_metadata.json"
    out_path = "outputs/graph_visualization.png"
    
    print("Loading graph structure...")
    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found.")
        return
        
    with open(graph_path, "r") as f:
        data = json.load(f)
        G = nx.node_link_graph(data)
        
    print("Loading spillover weights...")
    spillover_weights = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            spillover_weights = meta.get("spillover_weights", {})
            
    edges = G.edges()
    colors = []
    
    # Normalize spillover for colormap
    all_spill = list(spillover_weights.values())
    max_spillover = max(all_spill) if all_spill else 1.0
    if max_spillover == 0:
        max_spillover = 1.0
        
    for u, v in edges:
        pair_key = f"{u}|{v}"
        pair_key_rev = f"{v}|{u}"
        spillover = spillover_weights.get(pair_key, spillover_weights.get(pair_key_rev, 0.0))
        
        normalized_spill = min(1.0, spillover / max_spillover)
        colors.append(normalized_spill)
        
    print("Plotting graph...")
    plt.figure(figsize=(16, 12), facecolor='white')
    
    # Use spatial lat/lon layout if available
    pos = {}
    for node, attr in G.nodes(data=True):
        if 'lon' in attr and 'lat' in attr:
            pos[node] = (attr['lon'], attr['lat'])
            
    if len(pos) < len(G.nodes):
        pos = nx.spring_layout(G, seed=42)
        
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color='#3b82f6', alpha=0.9, edgecolors='white')
    
    # Red-Yellow-Green (reversed) so low (0) is green and high (1) is red
    edge_draw = nx.draw_networkx_edges(
        G, pos, edge_color=colors, edge_cmap=plt.cm.RdYlGn_r, edge_vmin=0, edge_vmax=1, width=2.5, alpha=0.8
    )
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", font_color="#1f2937")
    
    plt.title("Endogenous Network Graph & Historical Spillover Vectors", fontsize=18, fontweight='bold', pad=20)
    plt.suptitle("Green = Safe Diversion | Red = High Risk of Cascade Failure", fontsize=12, color='#4b5563')
    plt.axis("off")
    
    if edge_draw:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.5, pad=0.02)
        cbar.set_label("Spillover Severity", rotation=270, labelpad=20, fontsize=12)
        
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Graph visualization successfully saved to {out_path}")

if __name__ == "__main__":
    main()
