import os

# Define the directory structure
folders = {
    "data": ["sc-cell-line", "sc-tumor", "spatial", "bulk"],
    "preprocessed": ["sc-cell-line", "sc-tumor", "spatial", "bulk"]
}

# Create the folders
for parent, subfolders in folders.items():
    os.makedirs(parent, exist_ok=True)
    for subfolder in subfolders:
        os.makedirs(os.path.join(parent, subfolder), exist_ok=True)

print("Folders created successfully.")
