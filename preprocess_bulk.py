import pandas as pd
import numpy as np

gene_exp = pd.read_csv('./data/bulk/ALL_expression.csv', index_col=0)
labels = pd.read_csv('./data/bulk/ALL_label_binary_wf.csv', index_col=0)
labels = labels['CISPLATIN']

df = pd.concat([gene_exp, labels], axis=1)
df.to_csv('./preprocessed/bulk/bulk_data.csv')