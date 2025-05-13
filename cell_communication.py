import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt


def analyze_cell_communication(
    adata,
    pred_probs,
    cluster_threshold=0.5,
    sensitivity_key="predicted_sensitive",
    celltype_key= 'cell_type',
    spatial_neighbors_key="spatial_neighbors",
    ligrec_key="ligrec",
    figsize=(6, 5)
):
    """
    Perform cell-cell communication analysis on spatial transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression and spatial coordinates.
    pred_probs : array-like, shape (n_cells,)
        Predicted probability of sensitivity for each cell.
    cluster_threshold : float, optional (default: 0.5)
        Threshold to binarize pred_probs into sensitive vs resistant labels.
    sensitivity_key : str, optional
        Key under .obs to store binary sensitive/resistant labels.
    celltype_key : str or None, optional
        Key under .obs containing cell type annotations. If provided, analyses
        will include celltype-based interactions alongside sensitivity clusters.
    spatial_neighbors_key : str, optional
        Key under .obsp to store spatial neighbor graph.
    ligrec_key : str, optional
        Key under .uns where ligand-receptor results will be stored.
    figsize : tuple, optional
        Figure size for plotting.

    Returns
    -------
    None
    """

    # 1) Add predicted sensitivity labels
    sens_labels = (pred_probs >= cluster_threshold).astype(int)
    adata.obs[sensitivity_key] = pd.Categorical(sens_labels)

    # 2) Ensure cell type column exists if provided
    if celltype_key not in adata.obs:
        raise KeyError(f"Cell type key '{celltype_key}' not found in adata.obs")

    # 3) Compute spatial neighbors if missing
    if spatial_neighbors_key not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
        adata.obsp[spatial_neighbors_key] = adata.obsp.pop("spatial_connectivities")

    # 4) Run ligand-receptor analysis

    print(f"Running ligand–receptor analysis for '{celltype_key}'...")
    # Compute interactions, returns result dict
    result = sq.gr.ligrec(
        adata,
        cluster_key=celltype_key,
        use_raw=False,
        spatial_neighbors_key=spatial_neighbors_key,
        key_added=f"{ligrec_key}_{celltype_key}",
        copy=True
    )
    # Extract table
    if "metadata" in result and not result["metadata"].empty:
        lr_df = result["metadata"].copy()
    else:
        # Combine means & pvalues if available
        means = result.get("means")
        pvals = result.get("pvalues")
        if means is None or pvals is None or means.empty or pvals.empty:
            print(f"No ligand–receptor interactions found for '{celltype_key}'. Skipping plotting/saving.")
        lr_df = means.join(pvals, rsuffix="_pval").reset_index()
    # Save full interaction table
    outfile = f"output/ligrec_{celltype_key}.csv"
    lr_df.to_csv(outfile, index=False)
    print(f"Saved ligand–receptor interactions table to {outfile}")
    # Plotting: wrap in try to catch empty plots
    try:
        sq.pl.ligrec(
            result,
            title=f"Ligand–Receptor interactions ({celltype_key})",
            figsize=figsize,
            show=True
        )
        plt.show()
    except ValueError as e:
        print(f"Plotting skipped for '{celltype_key}': {e}")
    print("---")
