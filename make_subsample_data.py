import scanpy as sc
import pandas as pd
import os


spatial_data = sc.read("preprocessed/spatial/visium_breast_cancer.h5ad")  # Spatial graph data
bulk_data = pd.read_csv('preprocessed/bulk/bulk_data.csv')
sc_tumor_data = sc.read("preprocessed/sc-tumor/GSE169246.h5ad") # Single-cell tumor (no labels)
sc_cellline_data = sc.read("preprocessed/sc-cell-line/GSE131984.h5ad") # Cell line (drug labels)

common_genes = list(
    set(spatial_data.var_names) &
    set(bulk_data.columns) &
    set(sc_tumor_data.var_names) &
    set(sc_cellline_data.var_names)
)

spatial_data = spatial_data[:, common_genes]
bulk_data = bulk_data.loc[:, common_genes + ['PACLITAXEL']]
sc_tumor_data = sc_tumor_data[:, common_genes]
sc_cellline_data = sc_cellline_data[:, common_genes]


def subsample_adata(adata, fraction=0.1, random_state=42):
    """ Subsample an AnnData object. """
    sc.pp.subsample(adata, fraction=fraction, random_state=random_state)
    return adata


def subsample_dataframe(df, fraction=0.1, random_state=42):
    """ Subsample a pandas DataFrame. """
    return df.sample(frac=fraction, random_state=random_state)


fraction = 0.1  # Adjust based on memory constraints

print(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data)

spatial_data = subsample_adata(spatial_data, fraction)
bulk_data = subsample_dataframe(bulk_data, fraction)
sc_tumor_data = subsample_adata(sc_tumor_data, fraction)
sc_cellline_data = subsample_adata(sc_cellline_data, fraction)

print(spatial_data, bulk_data, sc_tumor_data, sc_cellline_data)

os.makedirs('subsampled/spatial', exist_ok=True)
os.makedirs('subsampled/bulk', exist_ok=True)
os.makedirs('subsampled/sc-tumor', exist_ok=True)
os.makedirs('subsampled/sc-cell-line', exist_ok=True)


spatial_data.write('subsampled/spatial/visium_breast_cancer.h5ad')
bulk_data.to_csv('subsampled/bulk/bulk_data.csv')
sc_tumor_data.write('subsampled/sc-tumor/GSE169246.h5ad')
sc_cellline_data.write("subsampled/sc-cell-line/GSE131984.h5ad")


