import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

spatial_data = sc.read(f"{'preprocessed'}/spatial/HumanBreastCancerPatient1_cropped.h5ad")

print(spatial_data)

# df = pd.read_csv('data/spatial/HumanBreastCancerPatient1_cell_metadata(1).csv')
# coord_x = df['center_x']
# coord_y = df['center_y']

# plt.scatter(coord_x, coord_y, s=0.001)
# print(len(coord_x))
# plt.show()