import os
import torch
import argparse
import warnings
import time
from utils import fix_seed, pca, mclust_R
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import scipy
from muon import atac as ac
from sklearn.neighbors import kneighbors_graph
import json

def main(args):
    
    # check whether GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # fix the random seed
    torch.cuda.empty_cache()
    fix_seed(args.random_seed)
    
    # load the two modality data, usually omic1 is RNA, omic2 is ADT(ATAC)
    from preprocess import read_preprocessed_data
    data = read_preprocessed_data(args)
    
    # define the trainer model
    from train import Train
    trainer = Train(args, device, data)
    
    # train the model and achieve latent representation and denoised results of both omics
    output = trainer.train()
    
    # add results to original datasets
    emb_combined = output['emb_combined']
    
   
    adata = data['adata_rna']
    adata.obsm['emb_combined'] = emb_combined
    
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="emb_combined")
    
    sc.tl.umap(adata)

    # mClust or kmeans
    adata.obsm['stGUMI_pca'] = pca(adata, use_reps="emb_combined", n_comps=20)
    adata = mclust_R(adata, num_cluster=args.n_clusters, used_obsm='stGUMI_pca')
    
    # UMAP
    sc.pl.umap(adata, color="mclust", title="Integration UMAP")
    plt.savefig(os.path.join(args.working_dir, 'Integration_UMAP.png'), dpi=300, bbox_inches='tight')

    sc.pl.embedding(adata, basis='spatial', color='mclust', title='Integration MClust Clusters', size=args.spot_size)
    plt.savefig(os.path.join(args.working_dir, 'Integtation_mClust_clusters.png'), dpi=300, bbox_inches='tight')
  

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Spatial Multi-omics Data Integration and Denoising')
    
    parser.add_argument("--config", type=str, help="Path to JSON config file. Can use either JSON file or 1by1 parameters setting.")
    
    parser.add_argument('--working_dir', type=str, default="/spatial/", help='Working Directory.')

    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed.') # 50

    parser.add_argument('--k_neighbor_graph', type=int, default=5, help='Parameter for k in k neighbor graph construction.')

    parser.add_argument('--out_feat', type=int, default=64, help='Output dimension for GCN and attention.')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.') # 1000
    
    # Plot
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for Mclust')
    parser.add_argument('--spot_size', type=int, default=20, help='Spot size of Spatial plot')
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # 使用 JSON 文件中的值更新 args 中的值
        for key, value in config.items():
            setattr(args, key, value)
            
    # print(args.mod_weight_list)
    main(args)