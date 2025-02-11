import scanpy as sc
import pandas as pd

spatial_data = sc.read("preprocessed/spatial/visium_breast_cancer.h5ad")  # Spatial graph data
bulk_data = pd.read_csv('preprocessed/bulk/bulk_data.csv')
sc_tumor_data = sc.read("preprocessed/sc-tumor/GSE169246.h5ad")  # Single-cell tumor (no labels)
sc_cellline_data = sc.read("preprocessed/sc-cell-line/GSE131984.h5ad")  # Cell line (drug labels)
spatial_data.var_names = spatial_data.var['gene_symbol']

# Extract common genes (adjust based on your data)
genes_dict = {
    'spatial': set(spatial_data.var_names),
    'bulk': set(bulk_data.columns),
    'cellline': set(sc_tumor_data.var_names),
    'tumor': set(sc_cellline_data.var_names)
}

genes_list = list(genes_dict.items())
for i in range(len(genes_list) - 1):
    type1, genes1 = genes_list[i]
    for j in range(i+1, len(genes_list)):
        type2, genes2 = genes_list[j]
        common = genes1.intersection(genes2)
        print(f'{type1} & {type2}: number comon genes: {len(common)}', common)
