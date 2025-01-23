import os
import argparse
import json
import warnings
from utils import fix_seed, pca, mclust_R
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import scipy
import anndata
import sklearn
import muon as mu
from muon import atac as ac

from matplotlib import rcParams
import warnings
from typing import Optional
warnings.filterwarnings('ignore')
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)

rcParams["figure.figsize"] = (4,4)

def main(args):
    
    # read raw data
    print(f"Reading {args.data_type} data!")
    atac_dir = os.path.join(args.data_dir, f"{args.data_type}_Signac.h5ad")
    adata = sc.read_h5ad(atac_dir) # ATAC
    adata.var_names_make_unique()
    
    # create save path
    plot_dir = os.path.join(args.output_dir, "Raw_Data_Plots") # save plot figures
    preprocessed_data_dir = os.path.join(args.output_dir, "Preprocessed_Data") # save preprocessed anndata
    if not os.path.exists(plot_dir): 
        os.makedirs(plot_dir)
    if not os.path.exists(preprocessed_data_dir): 
        os.makedirs(preprocessed_data_dir)
    
    # QC data
    
    # preprocess
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    
    print("Performing TFIDF normalization!")
    ac.pp.tfidf(adata, scale_factor=1e4)
    
    ac.tl.lsi(adata)
    adata.obsm['X_lsi'] = adata.obsm['X_lsi'][:,1:]
    adata.varm["LSI"] = adata.varm["LSI"][:,1:]
    adata.uns["lsi"]["stdev"] = adata.uns["lsi"]["stdev"][1:]
    
    # save anndata before select HVG
    print("Saving TFIDF_normalized h5ad!")
    adata.write_h5ad(os.path.join(preprocessed_data_dir, f"TFIDF_normalized_{args.data_type}.h5ad"))
    
    # Cluster
    sc.pp.neighbors(adata, use_rep = "X_lsi", n_neighbors=30, n_pcs=30)
    sc.tl.umap(adata)

    # mClust
    # adata.obsm['X_pca_pca'] = pca(adata, use_reps="X_lsi", n_comps=20)
    adata = mclust_R(adata, num_cluster=args.n_clusters, used_obsm='X_lsi')
    
    # UMAP
    sc.pl.umap(adata, color="mclust", title=f"Raw {args.data_type} UMAP", show=False)
    plt.savefig(os.path.join(plot_dir, f'raw_{args.data_type}_UMAP.png'), dpi=300, bbox_inches='tight')
    
    # spatial plot
    adata.obsm['spatial'][:,1] = -adata.obsm['spatial'][:,1]
    # adata.obsm['spatial'][:,0] = -adata.obsm['spatial'][:,0]
    
    sc.pl.embedding(adata, basis='spatial', color='mclust', title=f'Raw {args.data_type} MClust Clusters', size=args.spot_size, show=False)
    plt.savefig(os.path.join(plot_dir, f'raw_{args.data_type}_mClust_clusters.png'), dpi=300, bbox_inches='tight')
    

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess RNA.h5ad data, and perform raw data UMAP and spatial visualization.')
    
    # JSON file
    parser.add_argument("--config", type=str, help="Path to RNA JSON config file. Can use either JSON file or 1by1 parameters setting.")
    
    # IO
    parser.add_argument('--data_dir', type=str, default='/data/behmoaras/home/e1139777/scGUMI/Data/stereo_Liver_bin50', help='Dataset Directory.')
    parser.add_argument('--output_dir', type=str, default='/data/behmoaras/home/e1139777/stGUMI/Liver', help='Output path.')
    
    # Data type
    parser.add_argument('--data_type', type=str, default='ATAC', help='Just for Plot name. Please choose from ATAC, H3K27me3, H3K27ac, H3K4me3')
    
    # preprocess
    # parser.add_argument('--n_top_genes_for_HVG', type=int, default=3000, help='Select HVG. Parameters for sc.pp.highly_variable_genes()')
    parser.add_argument('--n_LSI', type=int, default=51, help='Dimension of LSI output')
    
    # Plot
    parser.add_argument('--n_clusters', type=int, default=15, help='Number of clusters for Mclust')
    parser.add_argument('--spot_size', type=int, default=20, help='Spot size of Spatial plot')
    
    # read json
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # 使用 JSON 文件中的值更新 args 中的值
        for key, value in config.items():
            setattr(args, key, value)
            
    # print(args.mod_weight_list)
    main(args)