import scipy.io
import scipy.sparse
import json
import numpy as np
import torch
import os
import networkx as nx
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def safe_to_list(x):
    """
    Safely convert numpy arrays, scalars, or other types to native Python lists/values,ensuring JSON serializability
    """
    if x is None:
        return None
    elif isinstance(x, (np.ndarray, np.generic)):
        return x.flatten().tolist()
    elif isinstance(x, (scipy.sparse.spmatrix,)):
        return x.tocoo().data.tolist()
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy().flatten().tolist()
    elif isinstance(x, (list, tuple)):
        return [safe_to_list(item) for item in x]
    elif isinstance(x, dict):
        return {k: safe_to_list(v) for k, v in x.items()}
    elif isinstance(x, (np.integer,)):
        return int(x)
    elif isinstance(x, (np.floating,)):
        return float(x)
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.cpu().numpy().tolist()
    else:
        return x  # Assume native Python types (int, float, str, bool)

def load_data_file(file_path):
    """Load .mat or .pt file"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.mat':
        try:
            data = scipy.io.loadmat(file_path)
            print(f" Successfully loaded .mat file: {file_path}")
            return 'mat', data
        except Exception as e:
            raise ValueError(f"Failed to load .mat file: {e}")

    elif ext == '.pt':
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            print(f" Successfully loaded .pt file: {file_path}")
            return 'pt', data
        except Exception as e:
            raise ValueError(f"Failed to load .pt file: {e}")
    else:
        raise ValueError("Unsupported file format, only .mat or .pt supported")

def process_data(data, file_type):
    """Process graph data from .mat or .pt file, extract unified information"""
    info = {}

    num_nodes = None
    edge_index = None
    edge_weight = None

    if file_type == 'mat':
        if 'Network' not in data:
            raise ValueError("Missing 'Network' field: adjacency matrix not found")
        network = data['Network']
        if not scipy.sparse.issparse(network):
            network = scipy.sparse.coo_matrix(network)
        else:
            network = network.tocoo()

        num_nodes = network.shape[0]
        row, col = network.row, network.col
        is_directed = not (np.array_equal(row, col[::-1]) and np.array_equal(col, row[::-1]))

        # Try to extract edge weights
        if 'edge_weight' in data:
            edge_weight = data['edge_weight'].flatten()
        elif hasattr(network, 'data'):
            edge_weight = network.data

    elif file_type == 'pt':
        if isinstance(data, dict):
            edge_index = data.get('edge_index') or data.get('adj')
            x = data.get('x') or data.get('feat') or data.get('features')
            y = data.get('y') or data.get('label') or data.get('labels')
            edge_weight = data.get('edge_attr')
            num_nodes = data.get('num_nodes')
        else:
            edge_index = getattr(data, 'edge_index', None)
            x = getattr(data, 'x', None)
            y = getattr(data, 'y', None)
            edge_weight = getattr(data, 'edge_attr', None)
            num_nodes = getattr(data, 'num_nodes', None)

        if edge_index is None:
            raise ValueError("Missing 'edge_index' field: edge information not found")

        edge_index = edge_index.cpu().numpy()
        row, col = edge_index[0], edge_index[1]

        if edge_weight is not None:
            edge_weight = edge_weight.cpu().numpy().flatten()

        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1

        edges_set = {(i, j) for i, j in zip(row, col)}
        reverse_set = {(j, i) for i, j in zip(row, col)}
        is_directed = not (edges_set == reverse_set)

    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Calculate degrees
    degrees = np.bincount(row, minlength=num_nodes)
    if not is_directed:
        degrees += np.bincount(col, minlength=num_nodes)

    info.update({
        'num_nodes': num_nodes,
        'num_edges_raw': len(row),
        'is_directed': is_directed,
        'degrees': degrees,
        'row': row,
        'col': col,
        'edge_weight': edge_weight
    })

    # Process features
    has_features = False
    num_features = 0
    feature_density = 0.0

    if file_type == 'mat':
        has_features = 'Attributes' in data
        if has_features:
            attrs = data['Attributes']
            if scipy.sparse.issparse(attrs):
                nnz = attrs.nnz
                num_features = attrs.shape[1]
                feature_density = nnz / (attrs.shape[0] * attrs.shape[1])
            else:
                num_features = attrs.shape[1]
                feature_density = np.count_nonzero(attrs) / attrs.size

    elif file_type == 'pt':
        has_features = (x is not None)
        if has_features:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            num_features = x.shape[1] if x.ndim > 1 else 1
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            nnz = np.count_nonzero(x)
            total = x.size
            feature_density = nnz / total

    info.update({
        'has_features': has_features,
        'num_features': num_features,
        'feature_density': feature_density
    })

    return info

def extract_graph_info(file_path):
    """Extract detailed graph information from .mat or .pt file and build JSON structure (completely remove label information)"""
    file_type, data = load_data_file(file_path)
    mat_info = process_data(data, file_type)

    num_nodes = mat_info['num_nodes']
    degrees = mat_info['degrees']
    is_directed = mat_info['is_directed']
    num_edges = mat_info['num_edges_raw'] if is_directed else mat_info['num_edges_raw'] // 2
    row = mat_info['row']
    col = mat_info['col']
    edge_weight = mat_info.get('edge_weight', None)

    graph_info = {
        "source": file_path,
        "file_type": file_type,
        "graph_structure": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "is_directed": bool(is_directed),
            "edge_density": float(num_edges / (num_nodes * (num_nodes - 1) / (2 if not is_directed else 1))),
        },
        "node_features": {
            "has_features": mat_info['has_features'],
            "num_features": mat_info['num_features'] if mat_info['has_features'] else 0,
            "feature_density": mat_info['feature_density'] if mat_info['has_features'] else 0.0,
            "sparsity_type": "sparse" if mat_info['has_features'] and isinstance(
                data.get('Attributes') if file_type == 'mat' else None, scipy.sparse.spmatrix
            ) else "dense"
        }
    }

    print(" Building NetworkX graph...")
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(row, col))
    if edge_weight is not None and len(edge_weight) == len(edges):
        edges = [(u, v, {'weight': w}) for (u, v), w in zip(edges, edge_weight)]
        G.add_edges_from(edges)
    else:
        G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    G_use = G if not is_directed else G.to_undirected()

    degree_entropy = -np.sum((degrees / degrees.sum() + 1e-10) * np.log2(degrees / degrees.sum() + 1e-10))
    graph_info["degree_analysis"] = {
        "degree_distribution": degrees.tolist(),
        "stats": {
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": float(degrees.mean()),
            "median": float(np.median(degrees)),
            "std": float(np.std(degrees)),
            "skewness": float(skew(degrees)),
            "kurtosis": float(kurtosis(degrees)),
            "entropy": float(degree_entropy)
        },
        "histogram_10bins": np.histogram(degrees, bins=10)[0].tolist(),
        "top_5_highest_degree_nodes": [
            {"node_id": int(idx), "degree": int(degrees[idx])}
            for idx in degrees.argsort()[-5:][::-1]
        ]
    }

    high_order = {}
    print(" Computing higher-order structure features...")

    try:
        triangles = nx.triangles(G_use)
        triangles_arr = np.array([triangles.get(i, 0) for i in range(num_nodes)])
        local_cc = nx.clustering(G_use)
        local_cc_arr = np.array([local_cc.get(i, 0.0) for i in range(num_nodes)])
        global_cc = nx.transitivity(G_use)
        high_order.update({
            'triangles_per_node': triangles_arr.tolist(),
            'avg_triangles_per_node': float(triangles_arr.mean()),
            'global_clustering_coefficient': global_cc,
            'clustering_coefficient_per_node': local_cc_arr.tolist(),
            'avg_clustering_coefficient': float(local_cc_arr.mean())
        })
    except Exception as e:
        print(f"    Triangle/clustering coefficient computation failed: {e}")
        high_order.update({
            'triangles_per_node': "computation_failed",
            'clustering_coefficient_per_node': "computation_failed"
        })

    try:
        core_numbers = nx.core_number(G_use)
        core_arr = np.array([core_numbers.get(i, 0) for i in range(num_nodes)])
        unique_cores, core_counts = np.unique(core_arr, return_counts=True)
        high_order.update({
            'core_number_per_node': core_arr.tolist(),
            'max_core_number': int(core_arr.max()) if len(core_arr) > 0 else 0,
            'core_distribution': {f"core_{int(k)}": int(v) for k, v in zip(unique_cores, core_counts)}
        })
    except Exception as e:
        print(f"    k-core computation failed: {e}")
        high_order['core_number_per_node'] = "computation_failed"

    try:
        deg_cent = nx.degree_centrality(G)
        deg_cent_arr = np.array([deg_cent.get(i, 0.0) for i in range(num_nodes)])
        high_order['degree_centrality'] = deg_cent_arr.tolist()
    except Exception as e:
        print(f"    Degree centrality computation failed: {e}")
        high_order['degree_centrality'] = "computation_failed"

    try:
        components = list(nx.connected_components(G_use)) if not is_directed else list(nx.weakly_connected_components(G))
        comp_sizes = [len(c) for c in components]
        comp_sizes.sort(reverse=True)
        high_order.update({
            'num_connected_components': len(components),
            'largest_component_size': comp_sizes[0] if comp_sizes else 0,
            'component_size_distribution_top10': comp_sizes[:10]
        })
    except Exception as e:
        print(f"    Connected components computation failed: {e}")
        high_order['num_connected_components'] = "computation_failed"

    try:
        close_cent = nx.closeness_centrality(G)
        close_cent_arr = np.array([close_cent.get(i, 0.0) for i in range(num_nodes)])
        high_order['closeness_centrality'] = close_cent_arr.tolist()
    except Exception as e:
        high_order['closeness_centrality'] = "computation_failed_or_skipped"


    try:
        k_sample = min(100, num_nodes)
        between_cent = nx.betweenness_centrality(G, k=k_sample, seed=42)
        between_cent_arr = np.array([between_cent.get(i, 0.0) for i in range(num_nodes)])
        high_order['betweenness_centrality'] = between_cent_arr.tolist()
    except Exception as e:
        high_order['betweenness_centrality'] = "computation_failed_or_skipped"


    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        high_order['degree_assortativity'] = assortativity
    except Exception as e:
        high_order['degree_assortativity'] = "computation_failed"

    if is_directed:
        try:
            reciprocity = nx.reciprocity(G)
            high_order['reciprocity'] = reciprocity
        except Exception as e:
            high_order['reciprocity'] = "computation_failed"

    try:
        if nx.is_connected(G_use):
            sample_nodes = list(G_use.nodes())[:min(100, num_nodes)]
            path_lengths = []
            for n in sample_nodes:
                lengths = nx.shortest_path_length(G_use, source=n)
                path_lengths.extend(lengths.values())
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            diameter = max(nx.eccentricity(G_use, v=sample_nodes).values())
        else:
            avg_path_length = "graph_disconnected"
            diameter = "graph_disconnected"
        high_order.update({
            'average_shortest_path_length': avg_path_length,
            'diameter': diameter
        })
    except Exception as e:
        high_order.update({
            'average_shortest_path_length': "computation_failed",
            'diameter': "computation_failed"
        })


    if edge_weight is not None and len(edge_weight) == len(row):
        try:
            high_order['edge_weight_analysis'] = {
                'mean': float(edge_weight.mean()),
                'std': float(edge_weight.std()),
                'min': float(edge_weight.min()),
                'max': float(edge_weight.max()),
                'histogram_10bins': np.histogram(edge_weight, bins=10)[0].tolist()
            }
        except Exception as e:
            high_order['edge_weight_analysis'] = "computation_failed"


    if not is_directed:
        try:
            rc = nx.rich_club_coefficient(G_use, normalized=False, seed=42)
            rc_sorted = sorted(rc.items())
            high_order['rich_club_coefficient_top10'] = {f"degree_{int(k)}": v for k, v in rc_sorted[-10:]}
        except Exception as e:
            high_order['rich_club_coefficient_top10'] = "computation_failed"


    try:
        communities = list(nx.community.greedy_modularity_communities(G_use))
        modularity = nx.community.modularity(G_use, communities)
        community_sizes = [len(c) for c in communities]
        community_sizes.sort(reverse=True)
        comm_entropy = -np.sum((np.array(community_sizes)/num_nodes + 1e-10) * np.log2(np.array(community_sizes)/num_nodes + 1e-10))
        high_order['community_structure'] = {
            'num_communities': len(communities),
            'modularity': modularity,
            'largest_community_size': community_sizes[0] if community_sizes else 0,
            'community_size_distribution_top5': community_sizes[:5],
            'community_entropy': float(comm_entropy)
        }
    except Exception as e:
        high_order['community_structure'] = "computation_failed"

    try:
        A = nx.adjacency_matrix(G_use).astype(np.float32)
        from scipy.sparse.linalg import eigsh

        eigenvals = eigsh(A, k=min(5, num_nodes-1), which='LM', return_eigenvectors=False)
        high_order['spectral_analysis'] = {
            'top_5_eigenvalues_adjacency': eigenvals.tolist(),
            'spectral_gap': float(eigenvals[-1] - eigenvals[-2]) if len(eigenvals) >= 2 else None
        }
    except Exception as e:
        high_order['spectral_analysis'] = "computation_failed"

    try:
        top_nodes = degrees.argsort()[-5:][::-1]
        ego_stats = []
        for nid in top_nodes:
            ego = nx.ego_graph(G_use, nid, radius=1)
            neighbors = list(ego.neighbors(nid)) if hasattr(ego, 'neighbors') else []
            neighbor_degrees = [degrees[n] for n in neighbors] if len(neighbors) > 0 else [0]
            ego_stats.append({
                "node_id": int(nid),
                "ego_size": len(ego.nodes()),
                "num_neighbors": len(neighbors),
                "avg_neighbor_degree": float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0,
                "std_neighbor_degree": float(np.std(neighbor_degrees)) if neighbor_degrees else 0.0
            })
        high_order['top_5_ego_network_stats'] = ego_stats
    except Exception as e:
        high_order['top_5_ego_network_stats'] = "computation_failed"

    graph_info["high_order_structure"] = high_order

    return graph_info

def save_graph_info_to_json(graph_info, output_path):
    """Save graph information to JSON file (safe serialization)"""
    try:
        cleaned = safe_to_list(graph_info)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        print(f" Successfully saved graph JSON information to: {output_path}")
    except Exception as e:
        print(f" Error saving JSON: {e}")
        raise

if __name__ == "__main__":
    input_file = "path/to/data/dataset_name.mat"  # Replace with your input file path
    output_json_file = "path/to/output/dataset_name.json"  

    try:
        print(f" Processing file: {input_file}")
        graph_info = extract_graph_info(input_file)
        save_graph_info_to_json(graph_info, output_json_file)

        # Print summary statistics
        print("\n Extraction complete, basic statistics:")
        print(f"   Number of nodes: {graph_info['graph_structure']['num_nodes']}")
        print(f"   Number of edges: {graph_info['graph_structure']['num_edges']}")
        print(f"   Feature dimensions: {graph_info['node_features']['num_features']}")

        ho = graph_info.get('high_order_structure', {})
        print(f"   Average triangles per node: {ho.get('avg_triangles_per_node', 'N/A')}")
        print(f"   Global clustering coefficient: {ho.get('global_clustering_coefficient', 'N/A')}")
        print(f"   Maximum k-core: {ho.get('max_core_number', 'N/A')}")
        print(f"   Number of connected components: {ho.get('num_connected_components', 'N/A')}")
        print(f"   Average closeness centrality: {np.mean(ho.get('closeness_centrality', [0])) if isinstance(ho.get('closeness_centrality'), list) else 'N/A'}")
        print(f"   Average betweenness centrality: {np.mean(ho.get('betweenness_centrality', [0])) if isinstance(ho.get('betweenness_centrality'), list) else 'N/A'}")
        print(f"   Degree assortativity: {ho.get('degree_assortativity', 'N/A')}")
        print(f"   Modularity: {ho.get('community_structure', {}).get('modularity', 'N/A') if isinstance(ho.get('community_structure'), dict) else 'N/A'}")

    except Exception as e:
        print(f" Error during processing: {e}")