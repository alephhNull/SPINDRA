import pandas as pd
import numpy as np


def preprocess_bulk(all_exp_file_name, all_label_file_name):
    gene_exp = pd.read_csv(f'data/bulk/{all_exp_file_name}', index_col=0)
    labels = pd.read_csv(f'data/bulk/{all_label_file_name}', index_col=0)
    labels = labels['PACLITAXEL']

    df = pd.concat([gene_exp, labels], axis=1)
    df.to_csv('./preprocessed/bulk/bulk_data.csv')

