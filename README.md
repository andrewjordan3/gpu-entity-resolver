# GPU-Accelerated Entity Resolver

A high-performance, modular Python package for resolving and deduplicating entities within a DataFrame. This project leverages the NVIDIA RAPIDS ecosystem (`cudf`, `cuML`, `cupy`) to execute the entire resolution pipeline on the GPU, enabling the processing of millions of records in minutes.

---

## Key Features

* **High-Performance GPU Pipeline**: The entire workflow, from text normalization to vectorization and clustering, runs on the GPU, minimizing CPU-GPU data transfer bottlenecks.
* **Multi-Stream Vectorization**: Creates rich entity embeddings by combining three distinct feature streams:
    * **TF-IDF**: Captures syntactic, character-level similarities.
    * **Phonetic**: Matches entities based on how they sound using the Metaphone algorithm.
    * **Semantic**: Understands the contextual meaning of entity names using Sentence-Transformers.
* **Advanced Clustering Ensemble**: Combines the precision of `HDBSCAN` with a high-recall, SNN-based graph clustering engine to accurately identify entity groups and rescue points that `HDBSCAN` might initially classify as noise.
* **Robust Validation & Merging**: Includes sophisticated, GPU-accelerated logic to validate cluster assignments, merge over-split clusters, and handle complex "chain" entities (e.g., the same company name at multiple addresses).
* **Modular and Configurable**: The entire pipeline is controlled by a single configuration object (`ResolverConfig`), making it easy to tune every parameter from text normalization rules to clustering hyperparameters.

---

## Installation

### 1. Prerequisites

This package requires the NVIDIA RAPIDS suite. The recommended way to install RAPIDS is via Conda. Please follow the official [RAPIDS installation guide](https://rapids.ai/start.html) to set up an environment with the necessary GPU libraries (`cudf`, `cuml`, etc.) that matches your system's CUDA version.

### 2. Install from Source

Once your Conda environment is activated, you can install this package directly from your local clone of the repository.

Navigate to the root directory of the project (where `setup.py` is located) and run the following command to install the package in editable mode:

```bash
pip install -e .
```
This will install the package and its CPU dependencies listed in `requirements.txt`.

## Basic Usage

Here is a simple example of how to use the EntityResolver to process a pandas DataFrame.

```python
import pandas as pd
from entity_resolver import EntityResolver, ResolverConfig

# 1. Load your data into a pandas DataFrame
# The DataFrame should have columns for entity names and addresses
data = {
    'company_name': [
        'Crystal Clean LLC', 'Crystal Clean', 'Crystal-Clean Inc.',
        'Midwest Waste Services', 'Midwest Waste'
    ],
    'address': [
        '123 Main St, Rockford IL', '123 Main Street, Rockford, IL', '123 Main St, Rockford',
        '456 Oak Ave, Hoffman Estates', '456 Oak Avenue, Hoffman Estates IL'
    ]
}
df = pd.DataFrame(data)

# 2. Initialize the configuration object
# You can customize any parameter here before passing it to the resolver
config = ResolverConfig()

# Example customization:
# config.modeling.hdbscan_params['min_cluster_size'] = 2
# config.validation.name_fuzz_ratio = 90

# 3. Initialize and run the resolver
resolver = EntityResolver(config)
resolved_df = resolver.fit_transform(df)

# 4. View the results
print(resolved_df[['company_name', 'canonical_name', 'confidence_score']])

# Expected Output:
#                  company_name        canonical_name  confidence_score
# 0           Crystal Clean LLC         Crystal Clean          0.95...
# 1               Crystal Clean         Crystal Clean          0.98...
# 2          Crystal-Clean Inc.         Crystal Clean          0.94...
# 3      Midwest Waste Services  Midwest Waste Services          0.97...
# 4               Midwest Waste  Midwest Waste Services          0.92...
```