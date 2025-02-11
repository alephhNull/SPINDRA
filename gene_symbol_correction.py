import scanpy as sc
import mygene


# Increase verbosity to see detailed logs
sc.settings.verbosity = 3

# Set figure parameters
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Load the .h5ad file
adata = sc.read_h5ad('data/spatial/visium-1142243F.h5ad')

# Inspect the data
print(adata)

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
print(adata)

adata.write('./preprocessed/spatial/symbol_corrected.h5ad')

