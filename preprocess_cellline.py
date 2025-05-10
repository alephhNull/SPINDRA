import scanpy as sc

def preprocess_celline(file_name):

    # 1. Load the data (keep)
    adata = sc.read_h5ad(f'data/sc-cell-line/{file_name}')
    print("Shape of Oringial Single-Cell cellline adata:", adata.shape)

    # 2. Quality Control (keep)
    adata.var['mt'] = adata.var_names.str.startswith('MT')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

    # Apply filters (keep)
    min_genes, max_genes, max_pct_mt = 200, 7000, 5
    adata = adata[adata.obs.n_genes_by_counts > min_genes, :]
    adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]

    if "_index" in adata.raw.var.columns:
        adata.raw.var.drop("_index", axis=1, inplace=True)
    
    print('Single-Cell Celline  adata final shape:', adata.shape)

    new_file_path = f'preprocessed/sc-cell-line/{file_name}'

    # 9. Save the preprocessed data
    adata.write(new_file_path)
    print(f'Preprocessed Single-Cell Cellline file saved at {new_file_path}')

