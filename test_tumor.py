import scanpy as sc

# 1. Load the data (keep)
adata = sc.read_h5ad('data/sc-tumor/gse169246.h5ad')
print("Shape of Oringial Single-Cell tumor adata:", adata.shape)

adata.layers["counts"] = adata.X.copy()
print(adata.X.max())

print(adata.var)

print(adata.obs['orig.ident'])



# 2. Quality Control (keep)
adata.var['mt'] = adata.var_names.str.startswith('MT')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

# just pretreatment selected
adata = adata[(adata.obs['orig.ident'] == 'Pre_P023_t') | 
              (adata.obs['orig.ident'] == 'Pre_P018_t')| 
              (adata.obs['orig.ident'] == 'Pre_P025_t')| 
              (adata.obs['orig.ident'] == 'Pre_P022_t')| 
              (adata.obs['orig.ident'] == 'Pre_P020_t'),:]

# Apply filters (keep)
min_genes, max_genes, max_pct_mt = 200, 7000, 5
adata = adata[adata.obs.n_genes_by_counts > min_genes, :]
adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]

if "_index" in adata.raw.var.columns:
    adata.raw.var.drop("_index", axis=1, inplace=True)

print('Single Cell Tumor adata final shape:', adata.shape)

sc.tl.pca(adata)

sc.pp.neighbors(adata)

sc.tl.umap(adata)

print(adata.obs['condition'])

sc.pl.umap(adata, color="orig.ident" , save= 'orig.iden_tumorsc.png')
sc.pl.umap(adata, color="condition" , save= 'condition_tumorsc.png')


sc.tl.leiden(adata, flavor="igraph",resolution=0.5)

sc.pl.umap(adata, color=["leiden"])
