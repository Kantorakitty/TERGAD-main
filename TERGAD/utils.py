
from torch_geometric.data import Data
import random
import os
import scipy.io as sio
import scipy.sparse as sp
import torch
import numpy as np
from sklearn.preprocessing import normalize
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from typing import Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve


def load_anomaly_detection_dataset(dataset, datadir='./data'):  
    """
    Load anomaly detection dataset
    """
    file_path = os.path.join(datadir, dataset)
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.mat':

        data_mat = sio.loadmat(file_path)
        
        print(f"Variables in .mat file: {[key for key in data_mat if not key.startswith('__')]}")
        
        if 'Network' in data_mat:
            adj = data_mat['Network']
        elif 'adj' in data_mat:
            adj = data_mat['adj']
        elif 'A' in data_mat:
            adj = data_mat['A']
        else:
            for key, value in data_mat.items():
                if not key.startswith('__') and hasattr(value, 'shape') and len(value.shape) == 2:
                    adj = value
                    print(f"Using variable '{key}' as adjacency matrix")
                    break
            else:
                raise ValueError("Adjacency matrix not found in .mat file")
        
        if 'Attributes' in data_mat:
            feat = data_mat['Attributes']
        elif 'attr' in data_mat:
            feat = data_mat['attr']
        elif 'X' in data_mat:
            feat = data_mat['X']
        elif 'feat' in data_mat:
            feat = data_mat['feat']
        else:
            # Try to find feature matrix
            for key, value in data_mat.items():
                if not key.startswith('__') and hasattr(value, 'shape') and len(value.shape) == 2:
                    feat = value
                    print(f"Using variable '{key}' as feature matrix")
                    break
            else:
                raise ValueError("Feature matrix not found in .mat file")
        
        if 'Label' in data_mat:
            truth = data_mat['Label']
        elif 'label' in data_mat:
            truth = data_mat['label']
        elif 'y' in data_mat:
            truth = data_mat['y']
        else:
            # Try to find labels
            for key, value in data_mat.items():
                if not key.startswith('__') and hasattr(value, 'shape') and len(value.shape) == 1:
                    truth = value
                    print(f"Using variable '{key}' as labels")
                    break
            else:
                raise ValueError("Labels not found in .mat file")
        
        truth = truth.flatten()
        
        # Convert to dense arrays
        if hasattr(adj, 'toarray'):
            adj = adj.toarray()
        if hasattr(feat, 'toarray'):
            feat = feat.toarray()
        
        # Normalize adjacency matrix
        adj_norm = normalize_adj(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
        adj_norm = adj_norm.toarray()
        
        # Original adjacency matrix (with self-loops)
        adj_original = adj + np.eye(adj.shape[0])

        return adj_norm, feat, truth, adj_original
        
    elif file_extension == '.pt':
    
        try:
            data_mat = torch.load(file_path, weights_only=True)
        except:
            data_mat = torch.load(file_path, weights_only=False)
        

        print(f"Keys in .pt file: {list(data_mat.keys())}")
        
        # Handle different .pt file formats
        if hasattr(data_mat, 'edge_index'):  # PyG Data object
            # Get adjacency matrix (constructed from edge_index)
            edge_index = data_mat.edge_index
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.cpu().numpy()
            
            num_nodes = data_mat.num_nodes if hasattr(data_mat, 'num_nodes') else int(edge_index.max()) + 1
            
            # Construct adjacency matrix
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                               (edge_index[0], edge_index[1])),
                               shape=(num_nodes, num_nodes)).toarray()
            
            # Get feature matrix
            if hasattr(data_mat, 'x'):
                feat = data_mat.x
            elif hasattr(data_mat, 'data.x'):
                feat = data_mat.data.x
            else:
                raise ValueError("Feature matrix not found in .pt file")
            
            # Get labels
            if hasattr(data_mat, 'y'):
                truth = data_mat.y
            elif hasattr(data_mat, 'data.y'):
                truth = data_mat.data.y
            else:
                raise ValueError("Labels not found in .pt file")
                
        else:  # Dictionary format
            # Get adjacency matrix
            if 'edge_index' in data_mat:
                edge_index = data_mat['edge_index']
                if isinstance(edge_index, torch.Tensor):
                    edge_index = edge_index.cpu().numpy()
                
                num_nodes = data_mat.get('num_nodes', int(edge_index.max()) + 1)
                adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                                   (edge_index[0], edge_index[1])),
                                   shape=(num_nodes, num_nodes)).toarray()
            elif 'adj' in data_mat:
                adj = data_mat['adj']
                if isinstance(adj, torch.Tensor):
                    adj = adj.cpu().numpy()
            else:
                raise ValueError("Adjacency matrix not found in .pt file")
            
            # Get feature matrix
            if 'data.x' in data_mat:
                feat = data_mat['data.x']
            elif 'x' in data_mat:
                feat = data_mat['x']
            elif 'feat' in data_mat:
                feat = data_mat['feat']
            else:
                raise ValueError("Feature matrix not found in .pt file")
            
            # Get labels
            if 'data.y' in data_mat:
                truth = data_mat['data.y']
            elif 'y' in data_mat:
                truth = data_mat['y']
            elif 'label' in data_mat:
                truth = data_mat['label']
            else:
                raise ValueError("Labels not found in .pt file")
        
        # Convert to numpy arrays
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        if isinstance(truth, torch.Tensor):
            truth = truth.cpu().numpy()
        
        truth = truth.flatten()
        
        # Normalize adjacency matrix
        adj_norm = normalize_adj(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
        adj_norm = adj_norm.toarray()
        
        # Original adjacency matrix (with self-loops)
        adj_original = adj + np.eye(adj.shape[0])
        
        return adj_norm, feat, truth, adj_original
         
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def l2_normalize_row(feat: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    l2_norm = np.linalg.norm(feat, axis=1, keepdims=True)
    l2_norm = np.where(l2_norm == 0, 1, l2_norm)  # 避免 0 向量除 0
    return feat / l2_norm

def l2_normalize_col(feat: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    l2_norm = np.linalg.norm(feat, axis=0, keepdims=True)
    l2_norm = np.where(l2_norm == 0, 1, l2_norm)  # 避免 0 向量除 0
    return feat / l2_norm
