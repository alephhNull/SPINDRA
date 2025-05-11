import pandas as pd
import numpy as np


def preprocess_bulk(all_exp_path, all_label_path):
    gene_exp = pd.read_csv(all_exp_path, index_col=0)
    labels = pd.read_csv(all_label_path, index_col=0)
    labels = labels['PACLITAXEL']

    df = pd.concat([gene_exp, labels], axis=1)
    df.to_csv('./preprocessed/bulk/bulk_data.csv')

