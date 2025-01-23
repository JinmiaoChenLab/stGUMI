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
    print("Reading ADT data!")
    adt_dir = os.path.join(args.data_dir, "ADT.h5ad")
    adata = sc.read_h5ad(adt_dir) # ADT
    adata.var_names_make_unique()
    
    # create save path
    plot_dir = os.path.join(args.output_dir, "Raw_Data_Plots") # save plot figures
    preprocessed_data_dir = os.path.join(args.output_dir, "Preprocessed_Data") # save preprocessed anndata
    if not os.path.exists(plot_dir): 
        os.makedirs(plot_dir)
    if not os.path.exists(preprocessed_data_dir): 
        os.makedirs(preprocessed_data_dir)
    
    # QC data
    # delete isotype or not
    if args.remove_isotype:
        print("Removing Isotype!")
        isotype_markers = [marker for marker in adata.var_names if marker.startswith('Isotype')]
        adata= adata[:, ~adata.var_names.isin(isotype_markers)]
    
    
    adata_rna = sc.read_h5ad(os.path.join(preprocessed_data_dir, "scaled_log_normalized_RNA.h5ad"))
    adata = adata[adata_rna.obs_names].copy()
    
    # preprocess
    print("Performing CLR Normalization!")
    adata = clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    
    # save anndata before select HVG
    print("Saving CLR normalized adt h5ad!")
    adata.write_h5ad(os.path.join(preprocessed_data_dir, "scaled_CLR_normalized_ADT.h5ad"))
    
    # PCA
    adata.obsm['X_pca'] = pca(adata, n_comps=args.n_PCA)
    
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    # mClust
    adata.obsm['X_pca_pca'] = pca(adata, use_reps="X_pca", n_comps=20)
    adata = mclust_R(adata, num_cluster=args.n_clusters, used_obsm='X_pca_pca')
    
    # UMAP
    sc.pl.umap(adata, color="mclust", title="Raw ADT UMAP", show=False)
    plt.savefig(os.path.join(plot_dir, 'raw_ADT_UMAP.png'), dpi=300, bbox_inches='tight')
    
    # spatial plot
    # adata.obsm['spatial'][:,1] = -adata.obsm['spatial'][:,1]
    # adata.obsm['spatial'][:,0] = -adata.obsm['spatial'][:,0]
    
    sc.pl.embedding(adata, basis='spatial', color='mclust', title='Raw ADT MClust Clusters', size=args.spot_size, show=False)
    plt.savefig(os.path.join(plot_dir, 'raw_ADT_mClust_clusters.png'), dpi=300, bbox_inches='tight')
    

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else adata.X)
    )
    return adata 

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess RNA.h5ad data, and perform raw data UMAP and spatial visualization.')
    
    # JSON file
    parser.add_argument("--config", type=str, help="Path to ADT JSON config file. Can use either JSON file or 1by1 parameters setting.")
    
    # IO
    parser.add_argument('--data_dir', type=str, default='/data/behmoaras/home/e1139777/scGUMI/Data/stereo_Liver_bin50', help='Dataset Directory.')
    parser.add_argument('--output_dir', type=str, default='/data/behmoaras/home/e1139777/stGUMI/Liver', help='Output path.')
    
    # QC
    parser.add_argument('--remove_isotype', action='store_true', default=False, help='Remove Isotype var_names in protein omics')
    
    # preprocess
    parser.add_argument('--n_PCA', type=int, default=50, help='Dimension of PCA output')
    
    # Plot
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for Mclust')
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