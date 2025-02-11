import scanpy as sc

adata = sc.read_h5ad('preprocessed/spatial/visium_breast_cancer.h5ad')
print(adata)
