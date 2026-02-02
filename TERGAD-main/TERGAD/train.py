from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import scipy.io
from datetime import datetime
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix   
import random

from utils import load_anomaly_detection_dataset
from sklearn.preprocessing import StandardScaler
from model import TERGAD

def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # Structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_dominant_dual(args):
    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    print(">>> Loading data...")

    # Load original attributes
    adj, attrs_source, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    print(f">>> Original attribute matrix shape: {attrs_source.shape}")

    # Load Embedding
    attrs_qwen = np.load(args.npy)
    print(f">>> Embedding matrix shape: {attrs_qwen.shape}")
    
    # Apply Z-score normalization to embedding
    scaler = StandardScaler()
    attrs_qwen = scaler.fit_transform(attrs_qwen)  # (N, D2)
    print("Embedding has been Z-score normalized!")

    # Verify node count consistency
    assert adj.shape[0] == attrs_source.shape[0] == attrs_qwen.shape[0], "Node count mismatch!"
    
    # Convert to PyTorch Tensor
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs_source = torch.FloatTensor(attrs_source)
    attrs_qwen = torch.FloatTensor(attrs_qwen)
    label = torch.LongTensor(label)  # Ensure label is LongTensor

    print("=" * 5 + " Starting  Training " + "=" * 5)

    # Model
    model = TERGAD(
        feat_size=attrs_source.size(1),
        qwen_feat_size=attrs_qwen.size(1),
        hidden_size=args.hidden_dim,
        dropout=args.dropout,
    )

    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs_source = attrs_source.to(device)
        attrs_qwen = attrs_qwen.to(device)
        label = label.to(device)
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    best_auc = 0.0
    best_cm = None

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()

        A_hat, X_hat = model(attrs_source, attrs_qwen, adj)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs_source, X_hat, args.alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()
        score = loss.detach().cpu().numpy()
        label_cpu = label.cpu().numpy()

        auc_score = roc_auc_score(label_cpu, score)
        anomaly_ratio = np.mean(label_cpu)

        # Threshold strategy
        threshold = np.percentile(score, 100 * (1 - anomaly_ratio))
        pred_labels = (score > threshold).astype(int)
       
        # Calculate other metrics
        accuracy = accuracy_score(label_cpu, pred_labels)
        precision_val = precision_score(label_cpu, pred_labels, zero_division=0)
        recall_val = recall_score(label_cpu, pred_labels, zero_division=0)

        
        # Evaluation
        print(f"Epoch: {epoch:04d}, "
              f"train_loss: {l.item():.5f}, "
              f"AUC: {auc_score:.5f}, "
              f"Accuracy: {accuracy:.5f}, "
              f"Precision: {precision_val:.5f}, "
              f"train/struct_loss: {struct_loss.item():.5f}, "
              f"train/feat_loss: {feat_loss.item():.5f}")

        # Update best metrics
        if auc_score > best_auc:
            best_auc = auc_score

    print(f"\n>>> Training completed!")
    print(f"Best AUC: {best_auc:.5f}")

# ============= Main Function =============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pubmed.mat', help='dataset name: pubmed.mat/citeseer.mat/acm.mat...')
    parser.add_argument('--npy', default='./bge_nodes_embedding/pubmed.npy', help='Path to Embedding .npy file')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter for loss')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()

    # ========== Global fixed random seed ==========
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> Global random seed set to: {seed}")
    # =============================================

    # Start training
    train_dominant_dual(args)