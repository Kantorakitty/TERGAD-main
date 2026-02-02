import torch
import torch.nn.functional as F
import json
import numpy as np
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List



# Directly return bge local model path
def find_model_path(model_name="bge-large-en-v1.5"):
    return "path/to/BAAI/bge-large-en-v1.5"
    

MODEL_PATH = find_model_path()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 8192

class BgeEmbedder:
    def __init__(self, model_path: str = MODEL_PATH):
        print(f"Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            local_files_only=True,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        print("Model loading complete!")

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        return last_hidden_states[
            torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
            attention_mask.sum(dim=1) - 1
        ]

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )

        return F.normalize(embeddings, p=2, dim=1).cpu().numpy() if normalize else embeddings.cpu().numpy()

def graph_info_to_node_texts_bge(graph_info: Dict) -> List[str]:

    gs = graph_info["graph_structure"]
    da = graph_info["degree_analysis"]
    ho = graph_info.get("high_order_structure", {})
    nf = graph_info.get("node_features", {})

    num_nodes = gs["num_nodes"]
    degrees = np.array(da["degree_distribution"])
    global_stats = da["stats"]
    global_avg = global_stats["mean"]
    global_med = global_stats["median"]
    global_max = global_stats["max"]
    global_std = global_stats["std"]

    triangles = np.array(ho.get("triangles_per_node", [0] * num_nodes))
    clustering_coeffs = np.array(ho.get("clustering_coefficient_per_node", [0.0] * num_nodes))
    core_numbers = np.array(ho.get("core_number_per_node", [0] * num_nodes))
    degree_centralities = np.array(ho.get("degree_centrality", [0.0] * num_nodes))
    closeness_centralities = np.array(ho.get("closeness_centrality", [0.0] * num_nodes))
    betweenness_centralities = np.array(ho.get("betweenness_centrality", [0.0] * num_nodes))


    community_structure = ho.get("community_structure", {})
    community_labels = [0] * num_nodes  # Default: no communities
    if isinstance(community_structure, dict) and "num_communities" in community_structure:
        try:
            communities = list(nx.community.greedy_modularity_communities(nx.Graph()))
            community_labels = np.zeros(num_nodes, dtype=int)
            for comm_id, comm_nodes in enumerate(communities):
                for node in comm_nodes:
                    if node < num_nodes:
                        community_labels[node] = comm_id
        except:
            pass

    top_5_hubs = [item["node_id"] for item in da["top_5_highest_degree_nodes"]]
    top_hub_set = set(top_5_hubs)

    dc_90 = np.percentile(degree_centralities, 90) if len(degree_centralities) > 0 else 0.0
    ccn_90 = np.percentile(closeness_centralities, 90) if len(closeness_centralities) > 0 else 0.0
    bc_90 = np.percentile(betweenness_centralities, 90) if len(betweenness_centralities) > 0 else 0.0
    deg_90 = np.percentile(degrees, 90) if len(degrees) > 0 else 0
    deg_95 = np.percentile(degrees, 95) if len(degrees) > 0 else 0
    deg_99 = np.percentile(degrees, 99) if len(degrees) > 0 else 0

    ego_stats_map = {}
    if "top_5_ego_network_stats" in ho and isinstance(ho["top_5_ego_network_stats"], list):
        for ego in ho["top_5_ego_network_stats"]:
            if isinstance(ego, dict) and "node_id" in ego:
                ego_stats_map[ego["node_id"]] = ego

    rich_club_top = ho.get("rich_club_coefficient_top10", {})

    spectral = ho.get("spectral_analysis", {})
    spectral_gap = spectral.get("spectral_gap", None)

    node_texts = []

    for node_id in range(num_nodes):
        deg = int(degrees[node_id]) if node_id < len(degrees) else 0
        tri = int(triangles[node_id]) if node_id < len(triangles) else 0
        cc = float(clustering_coeffs[node_id]) if node_id < len(clustering_coeffs) else 0.0
        core = int(core_numbers[node_id]) if node_id < len(core_numbers) else 0
        dc = float(degree_centralities[node_id]) if node_id < len(degree_centralities) else 0.0
        ccn = float(closeness_centralities[node_id]) if node_id < len(closeness_centralities) else 0.0
        bc = float(betweenness_centralities[node_id]) if node_id < len(betweenness_centralities) else 0.0
        comm_id = int(community_labels[node_id]) if node_id < len(community_labels) else 0

        is_top_hub = node_id in top_hub_set
        rank_in_top_hubs = None
        if is_top_hub:
            try:
                rank_in_top_hubs = top_5_hubs.index(node_id) + 1
            except ValueError:
                pass

        ego_stats = ego_stats_map.get(node_id, None)

        parts = []

        parts.append(f"Node {node_id} is a vertex in a {'directed' if gs['is_directed'] else 'undirected'} graph with {gs['num_nodes']} nodes and {gs['num_edges']} edges.")

        if deg == 0:
            parts.append("This node is isolated — it has zero connections to any other node.")
        else:
            parts.append(f"It has a degree of {deg}, which is:")
            if deg >= global_max:
                parts.append("the maximum degree in the entire graph.")
            elif deg >= deg_99:
                parts.append("in the top 1% of all nodes by degree (extremely high connectivity).")
            elif deg >= deg_95:
                parts.append("in the top 5% of all nodes by degree (very high connectivity).")
            elif deg >= deg_90:
                parts.append("in the top 10% of all nodes by degree (highly connected).")
            elif deg > global_avg + global_std:
                parts.append("above average plus one standard deviation (significantly connected).")
            elif deg > global_avg:
                parts.append("above the global average degree.")
            elif deg >= global_med:
                parts.append("at or above the global median degree.")
            elif deg > 0:
                parts.append("below the global median degree (relatively sparse connections).")

        if tri > 0:
            parts.append(f"This node participates in {tri} triangles, indicating it is part of tightly-knit local structures.")
        else:
            parts.append("This node participates in no triangles, suggesting its neighborhood lacks closure.")

        if cc >= 0.6:
            parts.append(f"It has a very high local clustering coefficient of {cc:.3f}, meaning nearly all its neighbors are interconnected.")
        elif cc >= 0.4:
            parts.append(f"It has a high clustering coefficient of {cc:.3f}, indicating strong local cohesiveness among its neighbors.")
        elif cc >= 0.2:
            parts.append(f"It has a moderate clustering coefficient of {cc:.3f}, suggesting some local structure.")
        elif cc > 0.0:
            parts.append(f"It has a low clustering coefficient of {cc:.3f}, indicating sparse interconnectivity among its neighbors.")
        else:
            parts.append("Its clustering coefficient is 0.0, meaning none of its neighbors are connected to each other.")

        if core > 0:
            parts.append(f"It resides in the k-core layer {core}, which means it remains connected even after iteratively removing all nodes with degree less than {core}.")
            if core >= ho.get("max_core_number", 0) * 0.8:
                parts.append("This places it among the most structurally robust nodes in the network.")
        else:
            parts.append("It resides in the 0-core, meaning it would be removed early in core decomposition due to low connectivity.")

        centrality_parts = []

        if dc > 0:
            centrality_parts.append(f"Degree centrality: {dc:.5f}")
            if dc >= dc_90:
                centrality_parts.append("(top 10% — highly influential in direct connectivity)")

        if ccn > 0:
            centrality_parts.append(f"Closeness centrality: {ccn:.5f}")
            if ccn >= ccn_90:
                centrality_parts.append("(top 10% — very close to all other nodes on average)")

        if bc > 0:
            centrality_parts.append(f"Betweenness centrality: {bc:.5f}")
            if bc >= bc_90:
                centrality_parts.append("(top 10% — critical bridge node between network regions)")

        if centrality_parts:
            parts.append("Centrality metrics: " + " | ".join(centrality_parts))


        parts.append(f"It belongs to community {comm_id} (modularity-based partitioning).")

        if ego_stats:
            parts.append(
                f"In its 1-hop ego network: it has {ego_stats['num_neighbors']} neighbors, "
                f"with average neighbor degree {ego_stats['avg_neighbor_degree']:.2f} ± {ego_stats['std_neighbor_degree']:.2f}, "
                f"and ego network size {ego_stats['ego_size']}."
            )

        if is_top_hub:
            if rank_in_top_hubs == 1:
                parts.append("This node is THE highest-degree hub in the entire graph — the most connected node.")
            elif rank_in_top_hubs <= 3:
                parts.append(f"This node is among the top {rank_in_top_hubs} most connected hubs in the graph.")
            else:
                parts.append(f"This node is ranked #{rank_in_top_hubs} among the top-5 connectivity hubs.")

        if rich_club_top and deg > 0:

            rc_keys = sorted([int(k.split('_')[1]) for k in rich_club_top.keys() if k.startswith('degree_')])
            for rc_deg in reversed(rc_keys):
                if deg >= rc_deg:
                    rc_val = rich_club_top.get(f"degree_{rc_deg}", 0.0)
                    parts.append(f"At degree threshold {rc_deg}, the rich club coefficient is {rc_val:.3f}, indicating {'strong' if rc_val > 0.5 else 'weak'} interconnectivity among high-degree nodes.")
                    break

        if nf.get("has_features", False):
            feat_dim = nf.get("num_features", 0)
            feat_density_raw = nf.get("feature_density", 0.0)

            if isinstance(feat_density_raw, list):
                feat_density = feat_density_raw[0] if len(feat_density_raw) > 0 else 0.0
            else:
                feat_density = float(feat_density_raw) if isinstance(feat_density_raw, (int, float, np.number)) else 0.0
            sparsity = "sparse" if nf.get("sparsity_type") == "sparse" else "dense"
            parts.append(f"This node is associated with a {feat_dim}-dimensional {sparsity} feature vector (feature density: {feat_density:.3f}).")


        if spectral_gap is not None:
            parts.append(f"Global spectral gap (Fiedler value) is {spectral_gap:.4f}, indicating {'strong' if spectral_gap > 0.1 else 'weak'} algebraic connectivity.")

        sentence = " ".join(parts).strip()
        if not sentence.endswith("."):
            sentence += "."
        sentence = " ".join(sentence.split())
        node_texts.append(sentence)

    return node_texts

def process_graph_data_nodes(input_json: str, output_npy: str):
    """
    Generate node-level embeddings
    """
    if not Path(input_json).exists():
        raise FileNotFoundError(f"Input file does not exist: {input_json}")
    
    with open(input_json) as f:
        data = json.load(f)

    # Generate description text for each node
    node_texts = graph_info_to_node_texts_bge(data)

    num_nodes = len(node_texts)
    
    print(f"Generating description text for {num_nodes} nodes...")
    
    # print("First 10 node descriptions:")
    # for i in range(min(10, num_nodes)):
    #     print(f"Node {i}: {node_texts[i]}")
    # print("="*50 + "\n")

    # Batch embed all nodes
    embedder = BgeEmbedder()
    
    # Process in batches to avoid memory overflow
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, num_nodes, batch_size):
        batch_texts = node_texts[i:i + batch_size]
        batch_embeddings = embedder.embed(batch_texts, normalize=False)   # Disable L2 normalization
        all_embeddings.append(batch_embeddings)
        print(f"Processing nodes {i}-{min(i+batch_size, num_nodes)-1}")
    
    # Combine all embeddings
    node_embeddings = np.vstack(all_embeddings)
    
    print(f"Node embedding matrix shape: {node_embeddings.shape}")
    np.save(output_npy, node_embeddings)
    print(f"Node embedding vectors saved to {output_npy}")
    return node_embeddings

if __name__ == "__main__":
    try:
        INPUT_JSON = "path/to/output/dataset_name.json"
        OUTPUT_NPY = "path/to/dataset_name_embeddings.npy"  # npy output path

        # Use node-level embedding
        emb = process_graph_data_nodes(INPUT_JSON, OUTPUT_NPY)
        print("Node-level embedding processing completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()