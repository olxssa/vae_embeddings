from corals.threads import set_threads_for_external_libraries
from corals.correlation.utils import derive_pvalues, multiple_test_correction
n_threads = 4
set_threads_for_external_libraries(n_threads)
# https://github.com/mgbckr/corals-lib-python/blob/main/docs/notebooks/full.ipynb
from corals.correlation.full.default import cor_full
import numpy as np
import scanpy as sc
import pandas as pd
from datetime import datetime

# Import data
adata = sc.read_h5ad("/Users/olyssa/PycharmProjects/VAE_embeddings/PBMC_adata.h5ad")

# Extract gene expression and embedding
gene_expression = adata.layers["counts"].toarray()
embedding = adata.obsm["embedding"]

# How many dimensions
n_embedding_features = embedding.shape[1]
n_gene_expression_features = gene_expression.shape[1]
n_samples = embedding.shape[0]

# Compute correlations
correlation_results = []
p_value_results = []
for feature_idx in range(n_embedding_features):
    start_time = datetime.now()
    print(f"Starting with embedding feature {feature_idx}")

    # Compute correlations
    feature_values = embedding[:, feature_idx].reshape(-1, 1)  # Ensure 2D shape
    correlations = cor_full(feature_values, gene_expression, method="spearman",
                            n_threads=n_threads)  # Compute all at once

    # Compute p-values
    p_values = derive_pvalues(correlations, n_samples).flatten()
    pvalues_corrected = multiple_test_correction(p_values, n_gene_expression_features, method="fdr_bh")

    correlation_results.append(correlations.flatten())
    p_value_results.append(pvalues_corrected)
    print(f"Finished with embedding feature {feature_idx}", datetime.now() - start_time)

# Convert to DataFrame and save
genes = adata.var_names
embedding_fts = np.concatenate([[i]*len(list(genes)) for i in range(n_embedding_features)])

df = pd.DataFrame({
    "Embedding_Feature": embedding_fts,
    "Gene": list(genes)*n_embedding_features,
    "Spearman_Correlation": np.concatenate(correlation_results),
    "P_Value": np.concatenate(p_value_results)
})

df.to_csv("corals_spearman_pvalues_results.csv", index=False)