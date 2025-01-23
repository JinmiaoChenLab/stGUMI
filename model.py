import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch.autograd import Function


# GCN Layer
class GCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(GCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)  # Linear transformation
        emb = torch.spmm(adj, x)  # Graph convolution
        
        return self.act(emb) if self.act else emb
    
    
# Attention Layer
class AttentionLayer(Module):
    """
    Attention layer for multi-modal fusion, supporting arbitrary number of modalities.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Parameters for attention mechanism
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, *emb_list):
        """
        Forward function for multi-modal attention.
        :param emb_list: Variable number of embeddings (N, in_feat) for each modality.
        :return: Fused embedding (N, in_feat) and attention scores (num_modalities,).
        """
        # Combine all embeddings along a new dimension
        emb_tensor = torch.stack(emb_list, dim=1)  # Shape: (N, num_modalities, in_feat)

        # Compute the attention scores
        v = F.tanh(torch.matmul(emb_tensor, self.w_omega))  # Shape: (N, num_modalities, out_feat)
        vu = torch.matmul(v, self.u_omega)  # Shape: (N, num_modalities, 1)
        alpha = F.softmax(vu.squeeze(-1), dim=1)  # Shape: (N, num_modalities)

        # Fuse embeddings using attention scores
        emb_combined = torch.sum(emb_tensor * alpha.unsqueeze(-1), dim=1)  # Shape: (N, in_feat)

        return emb_combined, alpha
    
    
# Multi-modal Encoder
class Encoder_omics(Module):
    """
    Multi-modal GCN encoder with attention-based fusion.
    """
    def __init__(self, args, dropout=0.0, act=None):
        super(Encoder_omics, self).__init__()

        self.args = args

        # Initialize GCN encoders for each modality
        self.encoder_omics = torch.nn.ModuleList([
            GCN(in_feat, args.out_feat, dropout=dropout, act=act)
            for in_feat in args.modalities_in_feats
        ])

        # Attention layer for cross-modal integration
        self.atten_cross = AttentionLayer(args.out_feat, args.out_feat)

        # Initialize GCN decoders for each modality
        self.decoder_omics = torch.nn.ModuleList([
            GCN(args.out_feat, out_feat, dropout=dropout, act=act)
            for out_feat in args.modalities_in_feats
        ])

    def forward(self, omics_list, adj_list):
        """
        :param omics_list: List of feature matrices for each modality. Shape: [(N, F1), (N, F2), ...]
        :param adj_list: List of adjacency matrices for each modality. Shape: [(N, N), (N, N), ...]
        :return: Tuple of encoded features, fused embedding, attention scores, and decoded outputs.
        """
        # Encoder: Process each modality independently
        embeddings_within = [
            encoder(omic, adj) for encoder, omic, adj in zip(self.encoder_omics, omics_list, adj_list)
        ]

        # Attention-based integration
        emb_combined, alpha_omics = self.atten_cross(*embeddings_within)

        # Decoder: Decode the fused embedding back to each modality
        decoded_outputs = [
            decoder(emb_combined, adj) for decoder, adj in zip(self.decoder_omics, adj_list)
        ]
        

        return embeddings_within, emb_combined, alpha_omics, decoded_outputs