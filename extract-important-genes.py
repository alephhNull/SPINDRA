import pandas as pd
import numpy as np


df = pd.read_csv('data/cosmic.csv')

genes = df[df['Tumour Types(Somatic)'].str.contains('breast', case=False, na=False)]['Gene Symbol']
print(list(genes))