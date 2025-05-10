import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


def perform_deg(spatial_data, original_adata_path):

    original_adata = sc.read(original_adata_path)

    # original_adata.var_names_make_unique()
    original_adata = original_adata[spatial_data.obs.index,:]


    # print(adata.shape)

    # adata = adata[adata.obs['cell_type'] == 'malignant cell',:]
    original_adata.obs['predicted_response'] = pd.Categorical(spatial_data.obs['predicted_response_prob'].map({0: 'Resistant', 1: 'Sensitive'}))

    adata = original_adata
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


    sc.tl.rank_genes_groups(adata, groupby="predicted_response", groups=['Resistant', 'Sensitive'], method="wilcoxon")
    deg_results = sc.get.rank_genes_groups_df(adata, group='Resistant')  # Results for group2 vs group1

    print(deg_results.head(50))
    deg_results.to_csv('deg_result_visium.csv', index=False)

    # Calculate -log10 adjusted p-values
    deg_results['-log10_pval'] = -np.log10(deg_results['pvals_adj'])

    # Define thresholds
    logfc_threshold = 1  
    pval_threshold = 0.05
    annotation_threshold = 10  # minimum -log10(p) for annotation

    # Determine significance based on thresholds
    deg_results['Significance'] = 'Not Significant'
    deg_results.loc[(deg_results['logfoldchanges'] > logfc_threshold) & 
                    (deg_results['pvals_adj'] < pval_threshold), 'Significance'] = 'Upregulated'
    deg_results.loc[(deg_results['logfoldchanges'] < -logfc_threshold) & 
                    (deg_results['pvals_adj'] < pval_threshold), 'Significance'] = 'Downregulated'

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=deg_results, x='logfoldchanges', y='-log10_pval', hue='Significance', 
                         palette={'Not Significant': 'gray', 'Upregulated': 'red', 'Downregulated': 'blue'}, 
                         alpha=0.6)

    # Draw threshold lines for reference
    plt.axvline(x=logfc_threshold, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=-logfc_threshold, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--', linewidth=0.5)

    # For each significant gene above the annotation threshold, draw a horizontal line and add label
    for _, row in deg_results.iterrows():
        if row['Significance'] in ['Upregulated', 'Downregulated'] and row['-log10_pval'] > annotation_threshold:
            x = row['logfoldchanges']
            y = row['-log10_pval']
            # Scale the horizontal line length based on the significance level.
            # For instance, more significant genes (larger y) get a longer horizontal line.
            # Here we use a simple scaling: offset = 0.2 * (-log10(p-value))
            offset = 0.2 * y  
            # Decide the direction: for upregulated genes, draw to the right; for downregulated, to the left.
            if row['logfoldchanges'] > 0:
                x_end = x + offset
            else:
                x_end = x - offset

            # Draw a horizontal dashed line from the gene's point to the computed label position.
            plt.hlines(y, x, x_end, colors='black', linestyles='dashed', linewidth=0.5)
            # Place the gene name at the end of the horizontal line.
            plt.text(x_end, y, row['names'], fontsize=8, va='center', ha='left' if row['logfoldchanges'] > 0 else 'right')

    # Set labels and title
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-log10 Adjusted P-value')
    plt.title('Volcano Plot: Sensitive vs Resistant')
    plt.legend(title='Significance')
    plt.show()

