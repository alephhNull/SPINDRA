import scanpy as sc


filename = 'GSE117872_HN120.h5ad'

# 1. Load the data (keep)
adata = sc.read_h5ad(f'data/sc-cell-line/{filename}')

print('shape before:', adata.shape)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 2. Quality Control (keep)
adata.var['mt'] = adata.var_names.str.startswith('MT')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)



sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)




# 9. Save the preprocessed data (keep)
adata.write(f'preprocessed/sc-cell-line/{filename}')