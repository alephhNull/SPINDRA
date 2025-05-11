import scanpy as sc
import mygene


def correct_gene_symbols(file_path, data_type):
    # Increase verbosity to see detailed logs
    sc.settings.verbosity = 3

    # Set figure parameters
    sc.settings.set_figure_params(dpi=80, facecolor='white')

    # Load the .h5ad file
    print(f"Loading the file from {file_path} for gene symbol correction...")
    adata = sc.read_h5ad(file_path)

    mg = mygene.MyGeneInfo()

    # Extract Ensembl IDs from var_names
    ensembl_ids = adata.var_names.str.upper()  # Ensure consistency

    # Query MyGene.info to get gene symbols
    query = ensembl_ids.tolist()
    results = mg.querymany(query, scopes='ensembl.gene', fields='symbol', species='human')

    # Create a mapping dictionary
    ensembl_to_symbol = {}
    for entry in results:
        if 'symbol' in entry and 'query' in entry:
            ensembl_to_symbol[entry['query'].upper()] = entry['symbol']

    # Add gene symbols to adata.var
    adata.var['gene_symbol'] = adata.var_names.map(ensembl_to_symbol)

    # Handle genes without a symbol
    missing = adata.var['gene_symbol'].isnull().sum()
    print(f"Number of genes without a mapped symbol: {missing}")

    # Optionally, remove genes without a symbol
    adata = adata[:, ~adata.var['gene_symbol'].isnull()].copy()
    # Ensure uniqueness and avoid conflicts
    adata.var_names = adata.var['gene_symbol'].astype(str)  # Assign as index

    # Drop the column since it's now redundant
    adata.var = adata.var.drop(columns=['gene_symbol'])
    adata.var_names_make_unique()

    print('Gene symbol correction done.')

    file_name = file_path.split('/')[-1].split('.')[0]
    new_file_name = file_name + '_symbol_corrected.h5ad'
    new_file_path = '/'.join(['preprocessed', data_type, new_file_name])

    adata.write(new_file_path)
    print(f'Gene corrected file saved at {new_file_path}')

    return adata
