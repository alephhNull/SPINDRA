import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gene_symbol_correction import correct_gene_symbols
import h5py

def preprocess_visium_spatial(file_path, library_id):
    adata = correct_gene_symbols(file_path, 'spatial')
    adata = sc.read_h5ad('preprocessed/spatial/GSM6592061_M15_symbol_corrected.h5ad')
    print("Shape of Oringial Visium Spatial adata:", adata.shape)

    adata = adata[adata.obs['in_tissue'] == 1,:]

    # 2. Quality Control
    adata.var['mt'] = adata.var_names.str.startswith('MT')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Visualize QC metrics
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    #              jitter=0.4, multi_panel=True)

    # Apply filters
    min_genes = 200
    max_genes = 7000
    max_pct_mt = 5

    adata = adata[adata.obs.n_genes_by_counts > min_genes, :]
    adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]

    # 3. Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4. Highly Variable Genes
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor='seurat_v3')
    # print(adata)

    # 5. Scale the data
    sc.pp.scale(adata, max_value=10)

    # 7. Handle Spatial Coordinates
    # if 'spatial' in adata.obsm.keys():
    #     sc.pl.spatial(adata, color='n_genes_by_counts', show=False, library_id=library_id)
    # else:
    #     print("Spatial coordinates not found in adata.obsm.")

    # Optional: Scale spatial coordinates
    adata.obsm['spatial_scaled'] = (adata.obsm['spatial'] - np.mean(adata.obsm['spatial'], axis=0)) / np.std(adata.obsm['spatial'], axis=0)

    # 8. Batch Correction (if necessary)
    if 'batch' in adata.obs.keys():
        sc.pp.combat(adata, key='batch')
        print("Batch correction applied using ComBat.")
    else:
        print("No batch information found; skipping batch correction.")


    print('Visium Spatial adata final shape:', adata.shape)
    
    file_name = file_path.split('/')[-1]
    new_file_path = f'preprocessed/spatial/{file_name}'
    adata.write(new_file_path)
    
    print(f'Preprocessed Visium spatial file saved at {new_file_path}')

    # 9. Save the preprocessed data

