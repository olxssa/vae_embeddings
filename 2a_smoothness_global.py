import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist
import concurrent.futures
import random
import pandas as pd
from datetime import datetime

# Import data
adata = sc.read_h5ad("PBMC_adata.h5ad")

# Define embedding space
embedding_data = adata.obsm["embedding"]

def compute_global_smoothness(Z):
    """
    Compute the global smoothness of the embedding space.
    """
    N = Z.shape[0]
    total_variance = 0
    for i in range(N):
        for j in range(i + 1, N):
            total_variance += np.linalg.norm(Z[i] - Z[j])**2

    return total_variance / (N * (N - 1))

# Compute global value
result = compute_global_smoothness(embedding_data)
g_time = datetime.now()
print("Lipschitz global computed successfully.")

# Convert results to NumPy array and save as CSV
pd.DataFrame([result], columns=["Global_smoothness"]).to_csv("smoothness_constant_global_all.csv", index=False)
print("Global smoothness constant saved successfully.")