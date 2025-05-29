import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import palantir

# 1. Load the data (keep)
adata = sc.read_h5ad('data/sc-cell-line/GSE131984.h5ad')
print("Shape of Oringial Single-Cell cellline adata:", adata.shape)

adata.layers["counts"] = adata.X.copy()
print(adata.X.max())

print(adata.var) 



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

sc.tl.pca(adata)

sc.pp.neighbors(adata)

sc.tl.umap(adata)

print(adata.obs['condition'])

sc.pl.umap(adata, color="orig.ident")


#sc.tl.leiden(adata, flavor="igraph",resolution=0.1)

#sc.pl.umap(adata, color=["leiden"])

sc.pl.umap(adata, color="condition")


# مرحله 5: محاسبه Pseudotime با Palantir
# تبدیل داده‌ها به فرمت مورد نیاز Palantir
pca_projections = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)

# استخراج نقشه‌های انتشار
ms_data = palantir.utils.determine_multiscale_space(dm_res)

# انتخاب سلول شروع (به صورت دستی یا بر اساس خوشه‌بندی)
start_cell = adata.obs_names[0]  # برای مثال، اولین سلول به عنوان شروع
pr_res = palantir.core.run_palantir(
    ms_data, start_cell, num_waypoints=500
)

# اضافه کردن pseudotime به adata
adata.obs['pseudotime'] = pr_res.pseudotime

# مرحله 6: مصورسازی نتایج
# ترسیم UMAP با رنگ pseudotime
sc.pl.umap(adata, color=['pseudotime', 'leiden'], save='_pseudotime.png')

# ترسیم pseudotime در یک نمودار پراکندگی
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=adata.obsm['X_umap'][:, 0],
    y=adata.obsm['X_umap'][:, 1],
    hue=adata.obs['pseudotime'],
    palette='viridis'
)
plt.title('Pseudotime on UMAP')
plt.savefig('pseudotime_umap.png')
plt.close()

print("تحلیل Pseudotime به پایان رسید.")








exit()
sc.tl.diffmap(adata)

print(adata.obsm['X_diffmap'])


# Setting root cell as described above
root_ixs = adata.obsm["X_diffmap"][:, 3].argmin()
sc.pl.scatter(
    adata,
    basis="diffmap",
    color=["clusters"],
    components=[2, 3],
)

adata.uns["iroot"] = root_ixs