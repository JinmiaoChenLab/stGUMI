import torch
from model import Encoder_omics
from utils import preprocess_graph, sparse_mx_to_torch_sparse_tensor
from torch import nn
import torch.nn.functional as F
from utils import plot_hist
#import pickle
from tqdm import tqdm
#import scipy.sparse as sp
import numpy as np
from scipy.sparse import coo_matrix
import random
from scipy import sparse
import matplotlib.pyplot as plt

class Train:
    
    def __init__(self, args, device, data):
        """
        Initialize the Train class with arguments, device, and data.
        
        :param args: Arguments for training and model configuration.
        :param device: Device to use for computation (CPU/GPU).
        :param data: Dictionary containing all modality AnnData objects.
        """
        
        self.args = args
        self.device = device
        self.data = data.copy()
        
        print("=== Initializing the Train class ===")
        
        # Extract all modalities from the data
        self.adata_list = [self.data[modality] for modality in self.data.keys()]  # 动态获取所有模态

        # Convert feature matrices to Torch tensors
        self.features_list = []
        for i, adata in enumerate(self.adata_list):
            if sparse.issparse(adata.X): 
                features = torch.FloatTensor(adata.X.toarray().astype(np.float32).copy()).to(self.device)
            else:
                features = torch.FloatTensor(adata.X.astype(np.float32).copy()).to(self.device)
            self.features_list.append(features)
            print(f"Converted modality {i + 1} features to Torch tensor with shape: {features.shape}")

        # Determine input feature dimensions for each modality
        self.args.modalities_in_feats = [features.shape[1] for features in self.features_list]
        
        # Extract spatial graph from the first modality
        self.graph = self.adata_list[0].obsm['graph_feat'].copy()

        # Ensure the graph is undirected
        self.graph = self.graph + self.graph.transpose()
        self.graph.data = np.ones(self.graph.data.shape)

        # Normalize the graph for training
        print("Normalized the graph for training.")
        self.adj = preprocess_graph(self.graph).to(self.device)

        # Create adjacency matrix list (shared graph for all modalities)
        self.adj_list = [self.adj for _ in self.adata_list]
        print("Created a shared adjacency matrix for all modalities.")
        
        # Give different modality different weights
        self.modalities_names = list(self.data.keys())
        self.args.mod_weight_list = []
        for modality_name in self.modalities_names:
            if "rna" in modality_name.lower():
                self.args.mod_weight_list.append(10)
            elif "adt" in modality_name.lower():
                self.args.mod_weight_list.append(1)
            elif "atac" in modality_name.lower():
                self.args.mod_weight_list.append(1)
            elif "h3k27me3" in modality_name.lower():
                self.args.mod_weight_list.append(1)
            elif "h3k27ac" in modality_name.lower():
                self.args.mod_weight_list.append(1)
            else:
                self.args.mod_weight_list.append(1)  # Default weight for all other modalities
        print(f"Assigned modality weights: {self.args.mod_weight_list}")
        
        
        print("=== Train class initialization completed successfully ===\n")
        
        
    def train(self):
        
        self.model = Encoder_omics(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate, 
                                          weight_decay=self.args.weight_decay)
        
        self.model.train()
        
        hist = [] # history for total
        hist_mods = [[] for _ in range(len(self.features_list))]  # Separate history for each modality
        
        for epoch in tqdm(range(self.args.epochs)):
        
            # Forward pass
            outputs = self.model(self.features_list, self.adj_list)
            embeddings_within, emb_combined, _, decoded_outputs = outputs
            
            # Compute reconstruction losses for each modality
            losses_mods = [
                F.mse_loss(self.features_list[i], decoded_outputs[i])
                for i in range(len(self.features_list))
            ]
            
            # Weighted sum of losses
            mod_weights = self.args.mod_weight_list
            loss = sum(mod_weights[i] * losses_mods[i] for i in range(len(self.features_list)))
        
            
            hist.append(loss.data.cpu().numpy())
            for i in range(len(self.features_list)):
                hist_mods[i].append(losses_mods[i].data.cpu().numpy())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

        # Plot loss history
        fig, axs = plt.subplots(len(self.features_list) + 1, 1, figsize=(5, 4 * (len(self.features_list) + 1)))

        plot_hist(hist, ax=axs[0], title="Total Loss")
        for i in range(len(self.features_list)):
            plot_hist(hist_mods[i], ax=axs[i + 1], title=f"Reconstruction Loss Modality {i + 1}")

        plt.tight_layout()
        # plt.savefig(self.args.output_dir + '/loss_histograms.png', dpi=300, bbox_inches='tight')
        
        print("=== Training finished! ===\n")    
        
        with torch.no_grad():
            self.model.eval()
            _, emb_combined, alpha_omics, decoded_outputs = self.model(self.features_list, self.adj_list)

        # Return decoded features, combined embedding, and attention weights
        return {
            "decoded_features": [output.detach().cpu().numpy() for output in decoded_outputs],
            "emb_combined": emb_combined.detach().cpu().numpy(),
            "alpha_omics": alpha_omics.detach().cpu().numpy(),
        }