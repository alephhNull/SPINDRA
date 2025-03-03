import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/spatial/HumanBreastCancerPatient1_cell_metadata(1).csv')
coord_x = df['center_x']
coord_y = df['center_y']

plt.scatter(coord_x, coord_y, s=0.001)
print(len(coord_x))
plt.show()