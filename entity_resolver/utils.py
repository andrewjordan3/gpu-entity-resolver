"""
A collection of helper and utility functions for the entity resolution pipeline.

This module contains functions for various tasks such as text processing,
address parsing, similarity calculations, graph operations, and vector
manipulations. These functions are designed to be pure and reusable,
decoupled from the main EntityResolver class state.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

# GPU/cuDF/cuML imports
import cudf
import cuml
import cupy
import cugraph
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.metrics.pairwise_distances import pairwise_distances
from cuml.neighbors import NearestNeighbors

# CPU/standard library imports
import phonetics
from postal.parser import parse_address
from postal.expand import expand_address

def get_canonical_name_gpu(name_series: cudf.Series) -> str:
    """
    Selects the best canonical name from a group using a GPU-accelerated
    scoring model based on centrality and descriptiveness.
    """
    if name_series.empty:
        return ""
    
    unique_names = name_series.unique()
    if len(unique_names) == 1:
        return unique_names.iloc[0]

    # Reset the index to ensure it is unique before passing to the vectorizer.
    # This prevents an internal reindexing error.
    unique_names_for_vec = unique_names.reset_index(drop=True)

    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    X = vec.fit_transform(unique_names_for_vec)
    
    sim_matrix = 1 - pairwise_distances(X, metric='cosine')
    
    name_counts = name_series.value_counts().reindex(unique_names).fillna(0)
    total_items = len(name_series)
    
    freq_weights = name_counts / total_items
    similarity_score = sim_matrix @ freq_weights.values
    
    base_score = similarity_score + freq_weights.values
    
    lengths = unique_names.str.len().astype('float32')
    length_bonus = cupy.log(lengths + 1).clip(max=3.5)
    
    final_scores = base_score * length_bonus
    
    return unique_names.iloc[final_scores.argmax()]

# --- Address Processing ---

def safe_parse_address(address_string: str, logger: logging.Logger) -> Dict:
    """
    Uses libpostal for robust address parsing. This is a CPU-bound function.
    """
    if not address_string or not isinstance(address_string, str):
        return {}
    try:
        # First, expand abbreviations (e.g., St -> Street).
        # expand_address returns a list of possibilities; we take the most likely one.
        expanded_list = expand_address(address_string)
        expanded_string = expanded_list[0] if expanded_list else address_string
        parsed = parse_address(expanded_string)
        parsed_components = {label: value for value, label in parsed}

        if 'po_box' in parsed_components:
            number_match = re.search(r'box\s*#?\s*(\d+)', expanded_string, re.IGNORECASE)
            street_name = f"PO BOX {number_match.group(1)}" if number_match else parsed_components.get('road', '')
            house_number = ''
        else:
            house_number = parsed_components.get('house_number', '')
            street_name = parsed_components.get('road', '')

        return {
            'address_line_1.street_number': house_number,
            'address_line_1.street_name': street_name,
            'city': parsed_components.get('city', ''),
            'state': parsed_components.get('state', '').upper(),
            'postal_code': parsed_components.get('postcode', '')[:5]
        }
    except Exception as e:
        logger.warning(f"libpostal parse error for '{address_string}': {e}")
        return {}
    
def create_address_key_gpu(gdf: cudf.DataFrame) -> cudf.Series:
    """
    A unified helper to create the normalized address key from its components.
    This serves as the single source of truth for the key's format.

    Args:
        gdf: A cuDF DataFrame that contains the component address columns
            (e.g., 'addr_street_number', 'addr_city', etc.).

    Returns:
        A cuDF Series containing the combined and normalized address key.
    """
    addr_cols = [
        'addr_street_number', 'addr_street_name', 'addr_city', 
        'addr_state', 'addr_zip'
    ]
    
    # Ensure all required columns exist, even if empty
    for col in addr_cols:
        if col not in gdf.columns:
            gdf[col] = ''

    key_components = [gdf[col].fillna('').astype(str) for col in addr_cols]
    
    # Concatenate all components into a single string
    normalized_key = key_components[0].str.cat(key_components[1:], sep=' ')
    
    # Apply final cleaning and return
    return normalized_key.str.lower().str.normalize_spaces()

def calculate_address_score_gpu(gdf: cudf.DataFrame) -> cudf.Series:
    """
    Calculates a completeness score for addresses on the GPU.

    This helper assigns weighted points based on the presence and validity
    of different address components. It's used to find the most
    descriptive address representation within a group.

    Args:
        gdf: A cuDF DataFrame containing address component columns
                (e.g., 'addr_street_name', 'addr_city').

    Returns:
        A cuDF Series of integer scores, with one score per row in the
        input DataFrame.
    """
    # Initialize a Series of zeros with the same index as the input DataFrame.
    # This ensures correct alignment.
    score = cudf.Series(0, index=gdf.index, dtype='int32')

    # --- Assign points for each component ---
    # The presence of a street name is the most important signal.
    score += (gdf['addr_street_name'].notna() & (gdf['addr_street_name'] != '')).astype('int32') * 2
    
    # Street number, city, and state are also important.
    score += (gdf['addr_street_number'].notna() & (gdf['addr_street_number'] != '')).astype('int32') * 1
    score += (gdf['addr_city'].notna() & (gdf['addr_city'] != '')).astype('int32') * 1
    score += (gdf['addr_state'].notna() & (gdf['addr_state'].str.len() == 2)).astype('int32') * 1
    
    # A valid 5-digit zip code adds to the score.
    is_valid_zip = (
        gdf['addr_zip'].notna() &
        (gdf['addr_zip'].str.len() == 5) &
        gdf['addr_zip'].str.isdigit()
    )
    score += is_valid_zip.astype('int32') * 1
    
    return score

def get_best_address_gpu(address_gdf: cudf.DataFrame) -> cudf.DataFrame:
    """
    Selects the single best address from a DataFrame of candidates on the GPU.

    It scores every unique address based on its completeness (presence of
    street number, name, city, state, zip) and uses frequency as a tie-breaker.
    This ensures the most descriptive and common address is chosen.

    Args:
        address_gdf: A cuDF DataFrame containing address records for a single group.
                    Must include 'addr_street_number', 'addr_street_name', etc.,
                    and 'addr_normalized_key'.

    Returns:
        A single-row cuDF DataFrame representing the best address, or an empty
        DataFrame if no valid address can be found.
    """
    if address_gdf.empty:
        return address_gdf

    # Get frequency of each unique address key within the group
    freq_map = address_gdf['addr_normalized_key'].value_counts()

    # Work with unique addresses to score them
    candidates = address_gdf.drop_duplicates(subset=['addr_normalized_key'])
    # Use a temporary index for merging frequency
    candidates = candidates.set_index('addr_normalized_key')
    candidates['freq'] = freq_map
    candidates = candidates.reset_index()

    # Calculate completeness score using vectorized operations
    candidates['score'] = calculate_address_score_gpu(candidates)

    # Sort by score (desc), then frequency (desc), then the key itself (for determinism)
    best_candidate = candidates.sort_values(
        ['score', 'freq', 'addr_normalized_key'], 
        ascending=[False, False, True]
    ).head(1)

    return best_candidate

# --- Similarity & Vector Calculations ---

def calculate_similarity_gpu(series_a: cudf.Series, series_b: cudf.Series, tfidf_params: Dict) -> cudf.Series:
    """
    Calculates the row-wise cosine similarity between two cuDF string series.
    """
    # Ensure consistent data types and handle potential nulls
    series_a = series_a.fillna('').astype(str)
    series_b = series_b.fillna('').astype(str)

    # Fit vectorizer with L2 normalization (default)
    combined_series = cudf.concat([series_a, series_b]).unique()
    vec = TfidfVectorizer(**tfidf_params)
    vec.fit(combined_series)

    # Transform to normalized vectors
    X_vec = vec.transform(series_a)
    Y_vec = vec.transform(series_b)

    # With L2-normalized vectors, cosine similarity = dot product!
    similarities = X_vec.multiply(Y_vec).sum(axis=1)
    similarities = cupy.asarray(similarities).flatten()
    
    return cudf.Series(similarities, index=series_a.index)

def balance_feature_streams(
    vector_streams: Dict[str, cupy.ndarray],
    proportions: Dict[str, float],
    eps: float = 1e-8
) -> List[cupy.ndarray]:
    """
    Balances the energy of feature streams to match target proportions.

    This method calculates the average energy (mean squared L2 norm) of
    each feature stream and then computes a scaling factor to adjust
    its energy to a desired proportion. This ensures that each stream
    contributes a controlled amount of variance to the final combined vector.

    Args:
        vector_streams: A dictionary where keys are stream names and values
                        are the dense CuPy arrays for that stream.
        proportions: A dictionary mapping stream names to their desired
                        final energy proportion (must sum to 1.0).
        eps: A small epsilon value to prevent division by zero.

    Returns:
        A list of the balanced (rescaled) CuPy arrays, ready for concatenation.
    """
    
    def block_energy(Z: cupy.ndarray) -> float:
        """Calculates the mean squared L2 norm for a dense matrix."""
        # .get() moves the final scalar value from GPU to CPU
        return float((Z * Z).sum(axis=1).mean().get())

    # Calculate the current energy of each stream
    energies = {name: block_energy(vec) for name, vec in vector_streams.items()}

    # Calculate the scaling factor 'a' for each stream where a = sqrt(p / m)
    scaling_factors = {
        name: (proportions[name] / (energies[name] + eps)) ** 0.5
        for name in vector_streams.keys()
    }

    # Apply the scaling factors
    balanced_vectors = [
        vector_streams[name] * scaling_factors[name]
        for name in vector_streams.keys()
    ]

    return balanced_vectors

# --- Graph & Clustering Utilities ---

def create_edge_list(indices: cupy.ndarray, distances: cupy.ndarray, threshold: float) -> cudf.DataFrame:
    """
    Creates a cuDF DataFrame of graph edges from the output of NearestNeighbors.

    Args:
        indices: The CuPy array of neighbor indices from kneighbors().
        distances: The CuPy array of neighbor distances from kneighbors().
        threshold: The distance threshold below which a pair is considered an edge.

    Returns:
        A cuDF DataFrame with 'source', 'dest', and 'distance' columns.
    """
    # Convert the output CuPy arrays to a DataFrame to create the edge list.
    # This avoids the index alignment error by creating flat series first.
    indices_cp = cupy.asarray(indices)
    distances_cp = cupy.asarray(distances)
    n_rows, n_neighbors = indices_cp.shape

    source_col = cudf.Series(indices_cp.flatten())
    distance_col = cudf.Series(distances_cp.flatten())
    dest_col = cudf.Series(cupy.arange(n_rows, dtype='int32')).repeat(n_neighbors).reset_index(drop=True)

    pairs = cudf.DataFrame({
        'source': source_col,
        'dest': dest_col,
        'distance': distance_col
    })
    
    # Filter for pairs that are below the similarity threshold and are not self-loops
    return pairs[(pairs['distance'] < threshold) & (pairs['source'] != pairs['dest'])]

def find_similar_pairs(
    string_series: cudf.Series,
    vectorizer_params: Dict[str, Any],
    nn_params: Dict[str, Any],
    similarity_threshold: float
) -> cudf.DataFrame:
    """
    A generic helper to find similar pairs within a series of strings.

    This method encapsulates the entire process of:
    1. Vectorizing the strings using TF-IDF.
    2. Building a NearestNeighbors model.
    3. Finding pairs of strings that are closer than the given threshold.

    Args:
        string_series: A cuDF Series of unique strings to compare.
        vectorizer_params: A dictionary of parameters for TfidfVectorizer.
        nn_params: A dictionary of parameters for NearestNeighbors.
        similarity_threshold: A float between 0 and 1. A lower value means
                                a stricter similarity requirement.

    Returns:
        A cuDF DataFrame with 'source' and 'dest' columns representing the
        indices of the matched pairs in the input string_series.
    """
    if len(string_series) < 2:
        return cudf.DataFrame({'source': [], 'dest': []})

    # Step 1: Vectorize the input strings
    vectorizer = TfidfVectorizer(**vectorizer_params)
    tfidf_matrix = vectorizer.fit_transform(string_series)

    # Step 2: Find nearest neighbors
    nn = NearestNeighbors(**nn_params)
    nn.fit(tfidf_matrix)
    distances, indices = nn.kneighbors(tfidf_matrix)

    # Step 3: Call the dedicated helper to create the edge list
    # The threshold is for distance, so a smaller value is stricter.
    matched_pairs = create_edge_list(
        indices=indices,
        distances=distances,
        threshold=similarity_threshold
    )
    
    # Return only the source and destination columns as required
    return matched_pairs[['source', 'dest']]

def find_graph_components(
    edge_list: cudf.DataFrame,
    source_col: str = 'source',
    dest_col: str = 'destination',
    vertex_col_name: str = 'vertex',
    component_col_name: str = 'component_id'
) -> cudf.DataFrame:
    """
    Finds connected components in a graph from an edge list using cugraph.

    This static helper encapsulates the boilerplate for creating a graph
    and running the weakly_connected_components algorithm.

    Args:
        edge_list: A cuDF DataFrame representing graph edges.
        source_col: The name of the source column in the edge_list.
        dest_col: The name of the destination column in the edge_list.
        vertex_col_name: The desired name for the output vertex column.
        component_col_name: The desired name for the output component ID column.

    Returns:
        A cuDF DataFrame with two columns: one for the vertex identifiers
        and one for their corresponding component ID.
    """
    if edge_list.empty:
        # Return an empty DataFrame with the correct columns if there are no edges
        return cudf.DataFrame({vertex_col_name: [], component_col_name: []})

    G = cugraph.Graph()
    G.from_cudf_edgelist(edge_list, source=source_col, destination=dest_col)

    components = cugraph.weakly_connected_components(G)
    
    # Rename columns to the desired generic output names
    return components.rename(columns={
        'labels': component_col_name,
        'vertex': vertex_col_name
    })

def build_mutual_rank_graph(vectors: cupy.ndarray, k: int) -> cugraph.Graph:
    """
    Builds a k-NN graph with a hybrid weighting scheme that combines
    mutual rank and cosine similarity.
    """
    # Find the k nearest neighbors and their distances for every point
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine').fit(vectors)
    distances, neighbor_indices = nn_model.kneighbors(vectors)
    
    # Convert cosine distances to similarities
    similarities = 1 - distances

    # Create a long-form DataFrame of all neighbor pairs (i -> j)
    source_nodes = cupy.repeat(cupy.arange(vectors.shape[0]), k)
    destination_nodes = neighbor_indices.ravel()
    # The rank is the position in the neighbor list (0 = closest)
    ranks = cupy.tile(cupy.arange(k), vectors.shape[0])
    
    # This DataFrame represents all directed edges (source -> destination)
    all_edges = cudf.DataFrame({
        'source': source_nodes,
        'destination': destination_nodes,
        'rank_of_dest_for_source': ranks,
        'similarity_of_dest_for_source': similarities.ravel()
    })
    
    # Filter out self-loops immediately
    all_edges = all_edges[all_edges['source'] != all_edges['destination']]

    # To find mutual neighbors, we merge the edge list with a swapped version
    edges_swapped = all_edges.rename(columns={
        'source': 'destination',
        'destination': 'source',
        'rank_of_dest_for_source': 'rank_of_source_for_dest',
        'similarity_of_dest_for_source': 'similarity_of_source_for_dest'
    })

    # This merge finds pairs that exist in both directions
    mutual_edges = all_edges.merge(edges_swapped, on=['source', 'destination'])
    
    # --- HYBRID WEIGHT CALCULATION ---
    # 1. Calculate the rank-based component (robust to density)
    rank_weight = 1.0 / (
        mutual_edges['rank_of_dest_for_source'] + 
        mutual_edges['rank_of_source_for_dest'] + 2
    )
    
    # 2. Calculate the similarity component (sensitive to distance)
    # We average the two directional similarities to keep it reciprocal
    similarity_weight = (
        mutual_edges['similarity_of_dest_for_source'] + 
        mutual_edges['similarity_of_source_for_dest']
    ) / 2.0
    
    # 3. Combine them: The final weight is high only if the points are
    #    both highly ranked AND very close.
    mutual_edges['weight'] = rank_weight * similarity_weight

    # Construct the final cugraph Graph object
    snn_graph = cugraph.Graph()
    # Set renumber=False to ensure vertex IDs are not changed,
    # which is critical for mapping the results back to the original data.
    snn_graph.from_cudf_edgelist(
        mutual_edges,
        source='source',
        destination='destination',
        edge_attr='weight',
        renumber=False
    )
    return snn_graph

# --- SNN Engine Helpers ---

def attach_noise_points(
        vectors: cupy.ndarray, 
        labels: cupy.ndarray, 
        k: int, 
        tau: float, 
        min_matching: int, 
        ratio_threshold: float
    ) -> cupy.ndarray:
    """
    A GPU-accelerated procedure to attach noise points to existing clusters
    if they have a strong, unambiguous connection, including a ratio test.
    """
    noise_indices = cupy.where(labels == -1)[0]
    if len(noise_indices) == 0:
        return labels

    # --- Query k+1 neighbors to account for self-inclusion ---
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(vectors)
    distances, neighbor_indices = nn_model.kneighbors(vectors[noise_indices])
    
    # --- Drop the first neighbor (the point itself) ---
    distances = distances[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]
    similarities = 1 - distances

    neighbor_labels = labels[neighbor_indices]
    final_labels = labels.copy()

    for i, original_noise_index in enumerate(noise_indices.tolist()):
        point_neighbor_labels = neighbor_labels[i]
        point_similarities = similarities[i]
        
        valid_neighbor_mask = point_neighbor_labels != -1
        if not valid_neighbor_mask.any():
            continue

        valid_labels = point_neighbor_labels[valid_neighbor_mask]
        valid_sims = point_similarities[valid_neighbor_mask]
        
        unique_candidate_labels, counts = cupy.unique(valid_labels, return_counts=True)
        # Sort candidates by count (most frequent first)
        sorted_indices = cupy.argsort(counts)[::-1]
        sorted_candidate_labels = unique_candidate_labels[sorted_indices]
        
        # Best candidate is the most frequent neighbor cluster
        best_candidate_label = sorted_candidate_labels[0]
        
        # --- Ratio Test Logic ---
        if len(sorted_candidate_labels) > 1:
            second_best_candidate_label = sorted_candidate_labels[1]
            
            # Calculate mean similarity to the best candidate's neighbors
            sims_to_best = valid_sims[valid_labels == best_candidate_label]
            mean_sim_to_best = sims_to_best.mean()

            # Calculate mean similarity to the second-best candidate's neighbors
            sims_to_second_best = valid_sims[valid_labels == second_best_candidate_label]
            mean_sim_to_second_best = sims_to_second_best.mean()

            # If the second best is too close, the assignment is ambiguous. Skip it.
            if mean_sim_to_best / (mean_sim_to_second_best + 1e-8) < ratio_threshold:
                continue
        
        # --- Original Strength Test (now applied only to the best candidate) ---
        similarities_to_candidate = valid_sims[valid_labels == best_candidate_label]
        meets_threshold_count = (similarities_to_candidate >= tau).sum()
        
        if (similarities_to_candidate.size >= min_matching and 
            similarities_to_candidate.mean() >= tau and 
            meets_threshold_count >= min_matching):
            final_labels[original_noise_index] = best_candidate_label
    
    return final_labels

def merge_snn_clusters(
        vectors: cupy.ndarray, 
        labels: cupy.ndarray,
        merge_params: Dict,
        logger: logging.Logger
    ) -> cupy.ndarray:
    """
    GPU-accelerated cluster merging with pre-filtering for efficiency
    """
    # Get all parameters 
    tau_med = merge_params['merge_median_threshold']
    tau_max = merge_params['merge_max_threshold']
    M = merge_params['merge_sample_size']
    batch_size = merge_params['merge_batch_size']
    similarity_threshold = merge_params['centroid_similarity_threshold']
    centroid_sample_size = merge_params['centroid_sample_size']

    unique_labels = cupy.unique(labels)
    clusters = [c for c in unique_labels.tolist() if c != -1]
    if len(clusters) < 2:
        return labels

    # Union-Find structure
    union_find_sets = {c: c for c in clusters}
    def find_set_root(cluster_id):
        while union_find_sets[cluster_id] != cluster_id:
            cluster_id = union_find_sets[cluster_id]
        return cluster_id
    def union_sets(c1, c2):
        root1 = find_set_root(c1)
        root2 = find_set_root(c2)
        if root1 != root2:
            union_find_sets[root1] = root2

    # Precompute cluster centroids for fast pre-filtering
    cluster_centroids = {}
    cluster_members = {}
    
    for c in clusters:
        members = cupy.where(labels == c)[0]
        cluster_members[c] = members
        # Sample for centroid calculation if cluster is large
        if len(members) > centroid_sample_size:
            sample_idx = cupy.random.choice(members, centroid_sample_size, replace=False)
            cluster_centroids[c] = cupy.mean(vectors[sample_idx], axis=0)
        else:
            cluster_centroids[c] = cupy.mean(vectors[members], axis=0)
    
    # Stack all centroids for batch processing
    centroid_matrix = cupy.vstack([cluster_centroids[c] for c in clusters])
    
    # Process clusters in batches to avoid memory issues
    n_clusters = len(clusters)
    merge_candidates = []
    
    for start_idx in range(0, n_clusters, batch_size):
        end_idx = min(start_idx + batch_size, n_clusters)
        batch_centroids = centroid_matrix[start_idx:end_idx]
        
        # Compute similarities between batch and all centroids
        similarities = 1 - pairwise_distances(
            batch_centroids, 
            centroid_matrix, 
            metric='cosine'
        )
        
        # Find pairs with high centroid similarity (pre-filter)
        high_sim_pairs = cupy.where(similarities > similarity_threshold)
        
        for i, j in zip(high_sim_pairs[0].tolist(), high_sim_pairs[1].tolist()):
            actual_i = start_idx + i
            if j > actual_i:  # Only check upper triangle
                merge_candidates.append((clusters[actual_i], clusters[j]))
    
    logger.info(f"Checking {len(merge_candidates)} candidate pairs (reduced from {n_clusters*(n_clusters-1)//2})")
    
    # Now only check the promising pairs in detail
    for cluster_1_id, cluster_2_id in merge_candidates:
        members_c1 = cluster_members[cluster_1_id]
        members_c2 = cluster_members[cluster_2_id]
        
        # Sample points
        if len(members_c1) > M:
            sample_indices1 = cupy.random.choice(members_c1, M, replace=False)
            vectors_c1 = vectors[sample_indices1]
        else:
            vectors_c1 = vectors[members_c1]

        if len(members_c2) > M:
            sample_indices2 = cupy.random.choice(members_c2, M, replace=False)
            vectors_c2 = vectors[sample_indices2]
        else:
            vectors_c2 = vectors[members_c2]

        # Compute detailed similarity
        distance_matrix = pairwise_distances(vectors_c1, vectors_c2, metric='cosine')
        similarity_matrix = 1 - distance_matrix
        
        if similarity_matrix.size > 0:
            median_similarity = float(cupy.median(similarity_matrix))
            max_similarity = float(cupy.max(similarity_matrix))
            
            if median_similarity >= tau_med and max_similarity >= tau_max:
                union_sets(cluster_1_id, cluster_2_id)

    # Relabel based on merged sets
    final_mapping = {c: find_set_root(c) for c in clusters}
    final_labels = labels.copy()
    for original_label, new_label in final_mapping.items():
        final_labels[labels == original_label] = new_label
        
    return final_labels

def normalize_rows(vectors: cupy.ndarray) -> cupy.ndarray:
    """
    Performs a safe, row-wise L2 normalization on a CuPy array.

    This ensures that each row vector has a magnitude of 1, which is essential
    for cosine similarity calculations where `cos(A, B) = dot(A_norm, B_norm)`.
    A small epsilon from the instance configuration is added to the norm to
    prevent division by zero for any all-zero row vectors.

    Args:
        vectors: A 2D CuPy array.

    Returns:
        The row-normalized CuPy array.
    """
    # Calculate the L2 norm (magnitude) for each row
    norms = cupy.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Use a small epsilon value to prevent division by zero
    return vectors / (norms + 1e-8)

def center_kernel_matrix(kernel_matrix: cupy.ndarray) -> cupy.ndarray:
    """
    Applies double-centering to a symmetric kernel (Gram) matrix.

    Double-centering is a critical step in Kernel PCA. It transforms the
    kernel matrix so that it represents the dot products of feature vectors
    that have been centered at the origin in the high-dimensional feature space.
    This is equivalent to performing PCA on centered data.

    The formula used is: K_centered = K - K_row_means - K_col_means + K_grand_mean

    Args:
        kernel_matrix: A square, symmetric CuPy array (e.g., a Gram matrix).

    Returns:
        The double-centered kernel matrix.
    """
    # Calculate the mean of each row and column, and the overall grand mean
    row_means = kernel_matrix.mean(axis=1, keepdims=True)
    col_means = kernel_matrix.mean(axis=0, keepdims=True)
    grand_mean = kernel_matrix.mean()

    # Apply the double-centering formula
    centered_kernel = kernel_matrix - row_means - col_means + grand_mean
    
    # Enforce perfect symmetry to counteract potential floating-point drift
    return (centered_kernel + centered_kernel.T) * 0.5

def center_kernel_vector(
    kernel_vector: cupy.ndarray,
    training_kernel_row_means: cupy.ndarray,
    training_kernel_grand_mean: float,
) -> cupy.ndarray:
    """
    Centers a new kernel vector against the statistics of the training kernel.

    This is the out-of-sample equivalent of `_center_kernel_matrix`. It projects
    new data points into the centered feature space defined by the original
    training data.

    The formula is: k_centered = k - training_row_means - k_col_means + training_grand_mean

    Args:
        kernel_vector: A (batch_size, n_samples) CuPy array of similarities
                    between new points and the original anchor points.
        training_kernel_row_means: The row means from the original training kernel matrix.
        training_kernel_grand_mean: The grand mean from the original training kernel.

    Returns:
        The centered kernel vector.
    """
    if kernel_vector.ndim == 1:
        # Ensure the vector is 2D for consistent matrix operations
        kernel_vector = kernel_vector[None, :]
    
    # Calculate the mean for each new row vector
    new_vector_row_means = kernel_vector.mean(axis=1, keepdims=True)
    
    # Apply the centering formula using the pre-computed training statistics
    # training_kernel_row_means.T broadcasts across the columns of kernel_vector
    centered_vector = (
        kernel_vector
        - training_kernel_row_means.T
        - new_vector_row_means
        + training_kernel_grand_mean
    )
    return centered_vector

def get_top_k_positive_eigenpairs(
    symmetric_matrix: cupy.ndarray,
    k: int,
    logger: logging.Logger
) -> Tuple[cupy.ndarray, cupy.ndarray]:
    """
    Calculates the top-k positive eigenpairs for a symmetric matrix.

    This function performs a full eigendecomposition, sorts the results in
    descending order, filters out any non-positive eigenvalues (which can occur
    due to numerical imprecision or non-PSD matrices), and returns exactly k
    eigenpairs, padding with zeros if necessary.

    Args:
        symmetric_matrix: The square, symmetric CuPy array to decompose.
        k: The desired number of eigenpairs to return.

    Returns:
        A tuple containing:
        - eigenvectors (cupy.ndarray): A (n, k) array of the top k eigenvectors.
        - eigenvalues (cupy.ndarray): A (k,) array of the top k eigenvalues.
    """
    # Add a warning if the requested number of components exceeds the matrix size.
    # The padding logic below will handle this case gracefully, but the user should be aware.
    if k > symmetric_matrix.shape[0]:
        logger.warning(
            f"Requested k={k} exceeds matrix size n={symmetric_matrix.shape[0]}; "
            "trailing dimensions will be zero-padded."
        )

    # Ensure the matrix is perfectly symmetric before decomposition
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) * 0.5
    
    # cupy.linalg.eigh is optimized for Hermitian (symmetric real) matrices
    # and returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = cupy.linalg.eigh(symmetric_matrix.astype(cupy.float32))
    
    # Sort eigenvalues and eigenvectors in descending order
    descending_indices = cupy.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_indices]
    eigenvectors = eigenvectors[:, descending_indices]
    
    # Filter for only numerically significant, positive eigenvalues
    positive_mask = eigenvalues > 1e-6
    if not cupy.any(positive_mask):
        logger.warning("No positive eigenvalues found. The kernel may be degenerate. Returning zeros.")
        return cupy.zeros((symmetric_matrix.shape[0], k), dtype=cupy.float32), cupy.zeros((k,), dtype=cupy.float32)
    
    positive_eigenvalues = eigenvalues[positive_mask]
    positive_eigenvectors = eigenvectors[:, positive_mask]
    
    # Ensure exactly k components are returned by padding or trimming
    num_positive = positive_eigenvectors.shape[1]
    if num_positive < k:
        pad_width = k - num_positive
        logger.debug(f"Found {num_positive} positive components; padding with {pad_width} zeros to reach k={k}.")
        
        # Pad eigenvectors with columns of zeros
        padded_eigenvectors = cupy.hstack([
            positive_eigenvectors,
            cupy.zeros((positive_eigenvectors.shape[0], pad_width), dtype=positive_eigenvectors.dtype)
        ])
        # Pad eigenvalues with zeros
        padded_eigenvalues = cupy.hstack([
            positive_eigenvalues,
            cupy.zeros((pad_width,), dtype=positive_eigenvalues.dtype)
        ])
        return padded_eigenvectors, padded_eigenvalues
    else:
        # Trim to the top k components
        return positive_eigenvectors[:, :k], positive_eigenvalues[:k]
