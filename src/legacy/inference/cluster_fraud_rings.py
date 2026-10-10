import os
import torch
import time
import pickle
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
PYG_GRAPH_PATH = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
EMBEDDINGS_SAVE_PATH = os.path.join(PROCESSED_DIR, "gnn_embeddings.pt")
NODE_MAPPING_PATH = os.path.join(PROCESSED_DIR, "node_mapping.pkl")

def find_fraud_rings():
    print("Loading components...")
    try:
        embeddings = torch.load(EMBEDDINGS_SAVE_PATH).numpy()
    except FileNotFoundError:
        print("❌ Error: Run extract_gnn_embeddings.py first!")
        return

    data = torch.load(PYG_GRAPH_PATH, weights_only=False)
    with open(NODE_MAPPING_PATH, 'rb') as f:
        node_mapping = pickle.load(f)
    
    # Reverse mapping: Integer ID -> Original Account String (e.g., C12345)
    reverse_mapping = {v: k for k, v in node_mapping.items()}

    # 1. Identify nodes involved in known fraud
    print("Identifying known fraud nodes from graph topology...")
    is_fraud_edge = data.edge_attr[:, 3].bool()
    fraud_edges = data.edge_index[:, is_fraud_edge]
    
    # Get unique node IDs that sent or received a fraudulent transaction
    known_fraud_nodes = torch.cat([fraud_edges[0], fraud_edges[1]]).unique().numpy()
    known_fraud_set = set(known_fraud_nodes)
    print(f"Found {len(known_fraud_set):,} accounts definitively linked to fraud.")

    # 2. Cluster the Embeddings
    # MiniBatchKMeans is used because standard KMeans crashes on 9 million rows
    n_clusters = 500  # Divide the network into 500 micro-neighborhoods
    print(f"\nClustering {embeddings.shape[0]:,} accounts into {n_clusters} network neighborhoods...")
    print("This may take 1-2 minutes...")
    
    start_time = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f"Clustering finished in {time.time() - start_time:.1f} seconds.")

    # 3. Analyze Clusters to find the "Rings"
    print("\nAnalyzing clusters for organized fraud rings...")
    cluster_fraud_counts = {}
    cluster_members = {}

    for node_id, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
            cluster_fraud_counts[cluster_id] = 0
            
        cluster_members[cluster_id].append(node_id)
        if node_id in known_fraud_set:
            cluster_fraud_counts[cluster_id] += 1

    # 4. Rank the most dangerous rings
    # Sort clusters by the highest concentration of known fraudsters
    sorted_clusters = sorted(cluster_fraud_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*60)
    print("🚨 TOP 5 SUSPECTED FRAUD RINGS DETECTED 🚨")
    print("="*60)
    
    for i in range(5):
        cluster_id, fraud_count = sorted_clusters[i]
        total_members = len(cluster_members[cluster_id])
        mule_suspects = total_members - fraud_count
        
        print(f"\nRing #{i+1} (Cluster ID: {cluster_id})")
        print(f"  - Total Accounts in Ring: {total_members:,}")
        print(f"  - Known Fraudsters: {fraud_count:,}")
        print(f"  - Suspected Mule Accounts: {mule_suspects:,} (Hidden Danger)")
        
        # Print a few example suspects to investigate
        suspect_ids = [n for n in cluster_members[cluster_id] if n not in known_fraud_set][:3]
        suspect_names = [reverse_mapping[n] for n in suspect_ids]
        print(f"  - Accounts to investigate immediately: {', '.join(suspect_names)}")

if __name__ == "__main__":
    find_fraud_rings()