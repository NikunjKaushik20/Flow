import pandas as pd
import numpy as np
import networkx as nx
import json
import os

DATA_PATH = "data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

def build_graph():
    print("=" * 60)
    print("BUILDING ENDOGENOUS GRAPH FROM ASTRAM DATASET")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Loading Astram dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Clean up identifiers
    df['corridor'] = df['corridor'].fillna("UNKNOWN").astype(str).str.strip().str.title()
    df['junction'] = df['junction'].fillna("UNKNOWN").astype(str).str.strip().str.title()
    
    # Only keep valid rows where both are known
    valid_mask = (df['corridor'] != "Unknown") & (df['junction'] != "Unknown") & \
                 (df['corridor'] != "Nan") & (df['junction'] != "Nan")
    df_valid = df[valid_mask].copy()
    
    print(f"Found {len(df_valid)} incidents with both corridor and junction metadata.")
    
    # Compute geographical centroids for UI map plotting
    # Convert lat/lon to numeric
    df_valid['latitude'] = pd.to_numeric(df_valid['latitude'], errors='coerce')
    df_valid['longitude'] = pd.to_numeric(df_valid['longitude'], errors='coerce')
    
    # Calculate centroids for corridors
    corridor_centroids = df_valid.groupby('corridor')[['latitude', 'longitude']].mean().dropna().to_dict('index')
    
    # Map Junction -> Set of Corridors
    junction_to_corridors = df_valid.groupby('junction')['corridor'].unique().to_dict()
    
    G = nx.Graph()
    
    # Add nodes with coordinates
    for corridor, coords in corridor_centroids.items():
        G.add_node(corridor, lat=coords['latitude'], lon=coords['longitude'])
        
    # Add edges based on shared junctions
    edge_count = 0
    for junction, corridors in junction_to_corridors.items():
        corridors = list(corridors)
        # Connect all corridors that meet at this junction
        for i in range(len(corridors)):
            for j in range(i + 1, len(corridors)):
                c1, c2 = corridors[i], corridors[j]
                if c1 in corridor_centroids and c2 in corridor_centroids:
                    # Calculate straight-line distance weight using coordinates
                    lat1, lon1 = corridor_centroids[c1]['latitude'], corridor_centroids[c1]['longitude']
                    lat2, lon2 = corridor_centroids[c2]['latitude'], corridor_centroids[c2]['longitude']
                    
                    # Haversine distance approx (km)
                    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0 
                    # Ensure minimum distance to avoid 0-weight edges
                    dist = max(0.1, dist)
                    
                    if G.has_edge(c1, c2):
                        # Keep the minimum distance if multiple junctions connect them
                        if dist < G[c1][c2]['weight']:
                            G[c1][c2]['weight'] = dist
                    else:
                        G.add_edge(c1, c2, weight=dist, junction=junction)
                        edge_count += 1
                        
    print(f"Graph Built: {G.number_of_nodes()} Corridors (Nodes), {G.number_of_edges()} Connections (Edges)")
    
    # Identify the largest connected component to ensure we have a viable routing network
    components = list(nx.connected_components(G))
    if components:
        largest_cc = max(components, key=len)
        print(f"Largest connected component contains {len(largest_cc)} corridors.")
        G_main = G.subgraph(largest_cc).copy()
    else:
        G_main = G

    # Save to JSON for the frontend and API
    graph_data = {
        "nodes": [{"id": node, "lat": data.get('lat'), "lon": data.get('lon')} for node, data in G_main.nodes(data=True)],
        "edges": [{"source": u, "target": v, "weight": data['weight'], "junction": data['junction']} for u, v, data in G_main.edges(data=True)]
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/graph_structure.json', 'w') as f:
        json.dump(graph_data, f, indent=4)
        
    print("\nGraph topology and coordinates saved to outputs/graph_structure.json")
    print("--> This topology was derived STRICTLY from the provided dataset!")

    # Test routing
    if G_main.number_of_nodes() > 10:
        nodes = list(G_main.nodes())
        import random
        random.seed(42) # For reproducible output
        # Find two distant nodes to test a path
        src, tgt = random.sample(nodes, 2)
        if nx.has_path(G_main, src, tgt):
            path = nx.shortest_path(G_main, src, tgt, weight='weight')
            print(f"\n[Test] Diversion Route from '{src}' to '{tgt}':")
            print(" -> ".join(path))

if __name__ == "__main__":
    build_graph()
