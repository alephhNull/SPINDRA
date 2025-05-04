import pandas as pd
import numpy as np
import scanpy as sc
import anndata

df = pd.read_csv('data/spatial/HumanBreastCancerPatient1_cell_by_gene.csv', index_col=0)
df_meta = pd.read_csv('data/spatial/HumanBreastCancerPatient1_cell_metadata.csv', index_col=0)

adata = anndata.AnnData(df)
adata.obsm['spatial'] = df_meta[['center_x', 'center_y']].values


adata = adata[:, ~adata.var_names.str.startswith("Blank-")]
square_filter = (adata.obsm['spatial'][:,0] > 4e+3) & (adata.obsm['spatial'][:,0] < 5e+3) & (adata.obsm['spatial'][:,1] > 3e+3) & (adata.obsm['spatial'][:,1] < 4e+3)
adata = adata[square_filter,:]

print("shape after filter:", adata.shape)
adata.write('data/spatial/HumanBreastCancerPatient1_cropped.h5ad')

# 2. Quality Control
adata.var['mt'] = adata.var_names.str.startswith('MT')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Visualize QC metrics
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

# # Apply filters
# min_genes = 200
# max_genes = 7000
# max_pct_mt = 5


# print('adata before', adata.shape)
# adata = adata[adata.obs.n_genes_by_counts > min_genes, :]
# adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
# adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]
# print('adata after', adata.shape)

# 3. Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


# 5. Scale the data
sc.pp.scale(adata, max_value=10)

# sc.tl.pca(adata, svd_solver='arpack')
# sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)

# 7. Handle Spatial Coordinates
if 'spatial' in adata.obsm.keys():
    sc.pl.spatial(adata, color='n_genes_by_counts', show=True, spot_size=10)
else:
    print("Spatial coordinates not found in adata.obsm.")

# Optional: Scale spatial coordinates
adata.obsm['spatial_scaled'] = (adata.obsm['spatial'] - np.mean(adata.obsm['spatial'], axis=0)) / np.std(adata.obsm['spatial'], axis=0)

adata.write('preprocessed/spatial/HumanBreastCancerPatient1_cropped.h5ad')