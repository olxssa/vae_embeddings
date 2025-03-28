import scanpy as sc
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
import random

# Import data
adata = sc.read_h5ad("PBMC_adata.h5ad")

# Define embedding space
embedding_data = adata.obsm["embedding"]

# Define original space
original_data = adata.layers["counts"].toarray()

# Reduce data
sample_size = 10**4
cell_indices = [random.randint(0, original_data.shape[0]-1) for c in range(0, sample_size)]

embedding_data_red = embedding_data[cell_indices]

atac = adata.obsm["ATAC"].toarray()[cell_indices]
original_data_red = original_data[cell_indices]
original_data_red_atac = np.concatenate((original_data_red, atac), axis=1)

# Function to compute pairwise Euclidean distance in parallel
def compute_distance(data):
    return squareform(pdist(data, metric='euclidean'))

# Compute distances in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_d_original = executor.submit(compute_distance, original_data_red_atac)
    future_d_embedding = executor.submit(compute_distance, embedding_data_red)

    d_original = future_d_original.result()
    d_embedding = future_d_embedding.result()
print("Distance matrices computed successfully.")

# Avoid division by zero (set minimum distance threshold)
epsilon = 1e-6
d_original[d_original < epsilon] = epsilon

# Function to compute Lipschitz constants for a batch of points
def compute_lipschitz_local(start, end):
    lipschitz_values = []
    for i in range(start, end):
        local_lipschitz = np.full_like(d_embedding[i], np.inf)  # Default to "inf"
        mask = d_original[i] > 0  # Identify valid divisions
        local_lipschitz[mask] = np.divide(d_embedding[i][mask], d_original[i][mask])
        lipschitz_values.append(np.max(local_lipschitz))
    return lipschitz_values

def compute_lipschitz_global(start, end):
    lipschitz_values = []
    for i in range(start, end):
        local_lipschitz = np.full_like(d_embedding[i], np.inf)  # Default to "inf"
        mask = d_original[i] > 0  # Identify valid divisions
        local_lipschitz[mask] = np.divide(d_embedding[i][mask], d_original[i][mask])
        lipschitz_values.append(np.max(local_lipschitz))
    return lipschitz_values

# Use multi-threading for parallel computation of Lipschitz constants
num_threads = 50
batch_size = embedding_data_red.shape[0] // num_threads
results_local = []

# Compute local values
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(compute_lipschitz_local, i * batch_size, (i + 1) * batch_size) for i in range(num_threads)]
    for future in concurrent.futures.as_completed(futures):
        results_local.extend(future.result())

# Convert results to NumPy array and save as CSV
lipschitz_constants_local = np.array(results_local)
pd.DataFrame(lipschitz_constants_local).to_csv("lipschitz_constants_local_reduced.csv", index=False)
print("Lipschitz constants saved successfully.")