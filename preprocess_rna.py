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
#from muon import atac as ac
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)

rcParams["figure.figsize"] = (4,4)

def main(args):
    
    # read raw data
    rna_dir = os.path.join(args.data_dir, "RNA.h5ad")
    adata = sc.read_h5ad(rna_dir) # RNA
    adata.var_names_make_unique()
    
    # create save path
    plot_dir = os.path.join(args.output_dir, "Raw_Data_Plots") # save plot figures
    preprocessed_data_dir = os.path.join(args.output_dir, "Preprocessed_Data") # save preprocessed anndata
    if not os.path.exists(plot_dir): 
        os.makedirs(plot_dir)
    if not os.path.exists(preprocessed_data_dir): 
        os.makedirs(preprocessed_data_dir)
    
    # QC data
    sc.pp.filter_cells(adata, min_genes=args.min_genes_for_filter_cell)
    sc.pp.filter_genes(adata, min_cells=args.min_cells_for_filter_gene)
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False
    )
    plt.savefig(os.path.join(plot_dir, 'RNA_QC_plot.png'), dpi=300, bbox_inches='tight')
    adata = adata[adata.obs.pct_counts_mt < args.max_pct_counts_mt, :].copy()
    # adata = adata[adata.obs.total_counts > 500, :].copy() # for liver only
    
    # preprocess
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_top_genes_for_HVG)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    
    # save anndata before select HVG
    # adata.write_h5ad(os.path.join(preprocessed_data_dir, "log_normalized_RNA.h5ad"))
    
    # select HVG
    adata =  adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)
    
    # save anndata before select HVG
    adata.write_h5ad(os.path.join(preprocessed_data_dir, "scaled_log_normalized_RNA.h5ad"))
    
    # PCA
    adata.obsm['X_pca'] = pca(adata, n_comps=args.n_PCA)
    
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    # mClust
    adata.obsm['X_pca_pca'] = pca(adata, use_reps="X_pca", n_comps=20)
    adata = mclust_R(adata, num_cluster=args.n_clusters, used_obsm='X_pca_pca')
    
    # UMAP
    sc.pl.umap(adata, color="mclust", title="Raw RNA UMAP", show=False)
    plt.savefig(os.path.join(plot_dir, 'raw_RNA_UMAP.png'), dpi=300, bbox_inches='tight')
    
    # spatial plot
    # adata.obsm['spatial'][:,1] = -adata.obsm['spatial'][:,1]
    # adata.obsm['spatial'][:,0] = -adata.obsm['spatial'][:,0]
    
    sc.pl.embedding(adata, basis='spatial', color='mclust', title='Raw RNA MClust Clusters', size=args.spot_size, show=False)
    plt.savefig(os.path.join(plot_dir, 'raw_RNA_mClust_clusters.png'), dpi=300, bbox_inches='tight')
    
    

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess RNA.h5ad data, and perform raw data UMAP and spatial visualization.')
    
    # JSON file
    parser.add_argument("--config", type=str, help="Path to RNA JSON config file. Can use either JSON file or 1by1 parameters setting.")
    
    # IO
    parser.add_argument('--data_dir', type=str, default='/data/behmoaras/home/e1139777/scGUMI/Data/stereo_Liver_bin50', help='Dataset Directory.')
    parser.add_argument('--output_dir', type=str, default='/data/behmoaras/home/e1139777/stGUMI/Liver', help='Output path.')
    
    # QC
    parser.add_argument('--min_genes_for_filter_cell', type=int, default=0, help='min_gene parameter for sc.pp.filter_cells()')
    parser.add_argument('--min_cells_for_filter_gene', type=int, default=3, help='min_cell parameter for sc.pp.filter_genes()')
    parser.add_argument('--max_pct_counts_mt', type=int, default=5, help='Filter bad quality cell based on pct_counts_mt')
    
    # preprocess
    parser.add_argument('--n_top_genes_for_HVG', type=int, default=3000, help='Select HVG. Parameters for sc.pp.highly_variable_genes()')
    parser.add_argument('--n_PCA', type=int, default=50, help='Dimension of PCA output')
    
    # Plot
    parser.add_argument('--n_clusters', type=int, default=22, help='Number of clusters for Mclust')
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