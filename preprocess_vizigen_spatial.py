import pandas as pd
import numpy as np
import scanpy as sc


df = pd.read_csv('data/spatial/HumanBreastCancerPatient1_cell_by_gene.csv')
df_meta = pd.read_csv('data/spatial/HumanBreastCancerPatient1_cell_metadata.csv')

print(df_meta)
