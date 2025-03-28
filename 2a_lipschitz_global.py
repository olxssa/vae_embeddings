import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist
import concurrent.futures
import random
import pandas as pd
from datetime import datetime

start_time = datetime.now()

# Import data
adata = sc.read_h5ad("PBMC_adata.h5ad")

# Define embedding space
embedding_data = adata.obsm["embedding"]

# Define original space
original_data = adata.layers["counts"].toarray()

### Reduce data to 10^4 cells
# Once at initialization
#sample_size = 10**4
#cell_indices = [random.randint(0, original_data.shape[0]-1) for c in range(0, sample_size)]
# For reproducibility
#cell_indices = pd.read_csv("/Users/olyssa/PycharmProjects/VAE_embeddings/2_Embedding_space_properties_investigation/2a_Stability_Analysis/cell_indices.csv")
#cell_indices = list(cell_indices["cell_indices"])

### If all cells should be used (no reduction)
embedding_data_red = embedding_data

atac = adata.obsm["ATAC"].toarray()#[cell_indices]
original_data_red = original_data#[cell_indices]
original_data_red_atac = np.concatenate((original_data_red, atac), axis=1)

# %%
# Function to compute pairwise Euclidean distance in parallel
def compute_distance(data):
    return pdist(data, metric='euclidean')

# Compute distances in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_d_original = executor.submit(compute_distance, original_data_red_atac)
    future_d_embedding = executor.submit(compute_distance, embedding_data_red)

    d_original = future_d_original.result()
    d_embedding = future_d_embedding.result()
d_time = datetime.now()
print("Distance matrices computed successfully:", start_time-d_time)

# Avoid division by zero (set minimum distance threshold)
epsilon = 1e-6
d_original[d_original < epsilon] = epsilon

# Function to compute Lipschitz constants for a batch of points
def compute_lipschitz_global():
    d_original[d_original == 0] = np.inf
    lipschitz_ratios = d_embedding / d_original
    return np.max(lipschitz_ratios)

# Compute local values
result = compute_lipschitz_global()
g_time = datetime.now()
print("Lipschitz global computed successfully:", d_time-g_time)

# Convert results to NumPy array and save as CSV
pd.DataFrame([result], columns=["Global_Lipschitz"]).to_csv("lipschitz_constant_global_reduced.csv", index=False)
print("Global lipschitz constant saved successfully.")