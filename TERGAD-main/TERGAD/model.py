import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution
import math


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, z_attr, z_qwen):
        # Gate weights
        gate_input = torch.cat([z_attr, z_qwen], dim=1)
        gate_weight = self.gate(gate_input)  # (N, H)

        # Fused representation
        z_fused = gate_weight * z_attr + (1 - gate_weight) * z_qwen
        return z_fused


class TERGADEncoder(nn.Module):
    def __init__(self, feat_size, qwen_feat_size, hidden_size):
        super(TERGADEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.gc_attr = GraphConvolution(feat_size, hidden_size)
        self.gc_qwen = GraphConvolution(qwen_feat_size, hidden_size)

        self.fusion_layer = GatedFusion(hidden_size)

        self.fuse_gc = GraphConvolution(hidden_size, hidden_size)

    def forward(self, x_source, x_qwen, adj):

        z_attr = F.relu(self.gc_attr(x_source, adj))   # (N, H)
        z_qwen = F.relu(self.gc_qwen(x_qwen, adj))     # (N, H)

        z_fused = self.fusion_layer(z_attr, z_qwen)   # (N, H)

        z_final = F.relu(self.fuse_gc(z_fused, adj))  # (N, H)

        return z_final
      


class TERGAD(nn.Module):
    def __init__(self, feat_size, qwen_feat_size, hidden_size, dropout=0.0):
        super(TERGAD, self).__init__()
        self.encoder = TERGADEncoder(feat_size, qwen_feat_size, hidden_size)
        self.decoder_attr = GraphConvolution(hidden_size, feat_size)
        self.dropout = dropout

    def forward(self, attrs_source, attrs_qwen, adj):
        z = self.encoder(attrs_source, attrs_qwen, adj)
        z = F.dropout(z, self.dropout, training=self.training)

        x_hat = self.decoder_attr(z, adj)
        a_hat = torch.sigmoid(torch.mm(z, z.t()))

        return a_hat, x_hat







































































