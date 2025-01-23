import os
import scanpy as sc
import scipy
import anndata
import sklearn
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional, Union
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 
from sklearn.decomposition import PCA
import muon as mu
from muon import atac as ac
# from utils import tfidf

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)

def read_preprocessed_data(args):

    folder_path = os.path.join(args.working_dir, "Preprocessed_Data") 

    file_dict = {
        "RNA": "scaled_log_normalized_RNA.h5ad",
        "ADT": "scaled_CLR_normalized_ADT.h5ad",
        "ATAC": "scaled_TFIDF_normalized_ATAC.h5ad",
        "H3K27me3": "TFIDF_normalized_H3K27me3.h5ad",
        "H3K27ac": "TFIDF_normalized_H3K27ac.h5ad",
        "H3K4me4": "TFIDF_normalized_H3K4me4.h5ad",
    }

    print("=== Checking and loading files ===")
    # create a dictionary to save data
    adata_dict = {}

    for modality, file_name in file_dict.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            print(f"data '{file_name}' exists，now reading {modality}...")
            adata_dict[f"adata_{modality.lower()}"] = sc.read_h5ad(file_path)
        else:
            print(f"data '{file_name}' does not exsit，skip {modality}.")
    print("=== File loading completed ===\n")
    
    
            
    # Align cells if at least one dataset is loaded
    if adata_dict:
        
#         print("=== Aligning cells across all modalities ===")
        
#         common_cells = None
#         for adata in adata_dict.values():
#             if common_cells is None:
#                 common_cells = adata.obs_names
#             else:
#                 # Compute intersection of cell names
#                 common_cells = np.intersect1d(common_cells, adata.obs_names)

#         # Subset each modality to retain only common cells
#         for key, adata in adata_dict.items():
#             adata_dict[key] = adata[common_cells].copy()
#             print(f"{key} aligned. Retained cell count: {adata_dict[key].n_obs}")

#         print("=== Cell alignment completed! ===\n")
        
        
        
#         print("=== Select the HVG and Scale the input ===")
        
#         for key, adata in adata_dict.items():
#             if "adt" not in key:  # Exclude ADT
#                 print(f"Processing {key} for HVG and scaling...")
#                 if "highly_variable" in adata.var.columns:
#                     print(f"Selecting highly variable genes for {key}.")
#                     adata = adata[:, adata.var.highly_variable]
#                 else:
#                     print(f"No HVG information found for {key}. Skipping HVG selection.")
                
#                 # Scale the data
#                 print(f"Scaling {key} data...")
#                 sc.pp.scale(adata, max_value=10)
#                 adata_dict[key] = adata.copy()
                
#         print("=== HVG selection and scaling completed ===\n")
        
        
        
        print("=== Calculating graph information ===")
        # Use spatial information from the first aligned modality to compute graph
        adata_sample = list(adata_dict.values())[0]  # Get the first aligned AnnData
        
        if 'spatial' in adata_sample.obsm:
            print("Calculating graph information...")
            graph_omics = kneighbors_graph(
                adata_sample.obsm['spatial'],
                n_neighbors=args.k_neighbor_graph,
                mode="connectivity",
                metric="minkowski",
                include_self=False,
                n_jobs=-1
            )
            print("Graph information calculated successfully!")

            # Add graph information to all modalities
            for key, adata in adata_dict.items():
                adata.obsm['graph_feat'] = graph_omics.copy()
                print(f"Graph information added to {key}")
            print("=== Graph calculation completed ===\n")
        else:
            print("Spatial information not found in any modality. Unable to compute graph.")
    else:
        print("No modality data loaded. Skipping alignment and graph computation.")
               
    # Display loaded data and their details
    print("=== Summary of loaded modalities ===")
    for key, adata in adata_dict.items():
        print(f"{key} accessed successfully：")
        print(adata)
        
    print("=== End of data reading ===")
        
    return adata_dict

