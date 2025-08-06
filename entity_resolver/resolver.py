"""
This module contains the primary EntityResolver class, which orchestrates the
entire GPU-accelerated entity resolution pipeline.
"""
import pandas as pd
import numpy as np
import re
import pickle
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# --- GPU-specific imports ---
try:
    import cudf
    import cuml
    from cuml.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from cuml.cluster import HDBSCAN
    from cuml.decomposition import PCA
    from cuml.manifold import UMAP
    from cuml.neighbors import NearestNeighbors
    import cupy
    import cugraph
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: cuDF/cuML/CuPy not found. GPU acceleration is disabled.")

# --- CPU libraries for specific tasks ---
import phonetics
from sentence_transformers import SentenceTransformer

# --- Local package imports ---
from .config import ResolverConfig
from .components import GPUTruncatedSVD
from . import utils

class EntityResolver:
    """
    A class to resolve and deduplicate entities using a GPU-accelerated pipeline.
    
    This resolver normalizes entity data, creates vector embeddings, clusters them
    to find matches, and provides a confidence score for each match.
    """
    def __init__(self, config: ResolverConfig):
        """
        Initializes the EntityResolver.

        Args:
            config: A ResolverConfig object containing all necessary parameters.
        """
        if not GPU_AVAILABLE:
            raise ImportError("This class requires cuDF, cuML, and CuPy for GPU acceleration.")
        
        self.config = config
        self.logger = self._setup_logger()
        
        # Attributes that will be fitted
        self.trained_encoders_: Dict[str, Any] = {}
        self.reduction_models_: Dict[str, Any] = {}
        self.canonical_map_: Optional[pd.DataFrame] = None
        self.resolved_gdf_: Optional[cudf.DataFrame] = None
        self.cluster_model_: Optional[cuml.cluster.HDBSCAN] = None 
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> 'EntityResolver':
        """
        Fits the resolver pipeline on the provided data. This includes training
        encoders and building the clustering model.

        Args:
            df: A pandas DataFrame containing the entity data.

        Returns:
            The fitted EntityResolver instance.
        """
        self.logger.info(f"=== Starting Training on {len(df)} records ===")
        gdf = self._prepare_gpu_dataframe(df)
        self._process_pipeline(gdf, is_training=True)
        self._is_fitted = True
        self.logger.info("=== Training Complete ===")
        if self.canonical_map_ is not None:
            self.logger.info(f"Canonical map with {len(self.canonical_map_)} clusters built.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using the already fitted resolver. It assigns new
        records to existing clusters (canonical entities).

        Args:
            df: A pandas DataFrame with new records to resolve.

        Returns:
            A pandas DataFrame with resolved entity information.
        """
        if not self._is_fitted:
            raise RuntimeError("Resolver has not been fitted. Please call fit() first.")
        self.logger.info(f"=== Starting Transformation of {len(df)} new records ===")
        gdf = self._prepare_gpu_dataframe(df)
        resolved_gdf = self._process_pipeline(gdf, is_training=False)
        self.resolved_gdf_ = resolved_gdf
        # Convert the final result back to a pandas DataFrame for the user
        resolved_df = resolved_gdf.to_pandas()
        return self._split_canonical_address(resolved_df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A convenience method that fits the resolver on the data and then
        transforms the same data.

        Args:
            df: A pandas DataFrame to fit and then resolve.

        Returns:
            A pandas DataFrame with resolved entity information.
        """
        self.logger.info(f"=== Starting Fit & Transform on {len(df)} records ===")
        gdf = self._prepare_gpu_dataframe(df)
        resolved_gdf = self._process_pipeline(gdf, is_training=True)
        self._is_fitted = True
        self.logger.info("=== Fit & Transform Complete ===")
        self.resolved_gdf_ = resolved_gdf
        # Convert the final result back to a pandas DataFrame for the user
        resolved_df = resolved_gdf.to_pandas()
        return self._split_canonical_address(resolved_df)

    # --- Pipeline Step Implementations ---

    def normalize_text(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Normalizes entity names using a GPU-accelerated process.

        This method operates on a cuDF DataFrame, applies a series of string
        normalization rules, and creates a new 'normalized_text' column.

        Args:
            gdf: The input cuDF DataFrame.

        Returns:
            The cuDF DataFrame with an added 'normalized_text' column.
        """
        self.logger.debug(f"Normalizing text in '{self.config.columns.entity_col}' column...")
        
        # The input is already a cuDF DataFrame, so we can work on the Series directly.
        normalized_series = gdf[self.config.columns.entity_col].fillna('').astype(str)

        # --- Apply Normalization Steps Sequentially using cuDF's vectorized methods ---
        # 1. Basic cleaning: lowercase, handle common separators
        normalized_series = normalized_series.str.lower()
        normalized_series = normalized_series.str.replace('&', ' and ', n=-1)
        normalized_series = normalized_series.str.replace('+', ' and ', n=-1)

        # 2. Remove content in parentheses (e.g., "(some text)")
        normalized_series = normalized_series.str.replace(r'\([^)]*\)', '')

        # 3. Handle DBA/FKA/AKA separators, keeping only the text that comes after.
        dba_pattern = r'(?:\s|^)(?:d(?:/|\s)?b(?:/|\s)?a|f(?:/|\s)?k(?:/|\s)?a|a(?:/|\s)?k(?:/|\s)?a)\s+(.*)'
        extracted_names = normalized_series.str.extract(dba_pattern)
        # Where the pattern matched, use the extracted name; otherwise, keep the original.
        normalized_series = extracted_names[0].fillna(normalized_series)

        # 4. Apply custom word replacements from config
        for old, new in self.config.normalization.replacements.items():
            pattern = r'\b' + re.escape(old) + r'\b'
            normalized_series = normalized_series.str.replace(pattern, new)

        # 5. Remove legal suffixes from config.
        if self.config.normalization.suffixes_to_remove:
            suffix_pattern = r'\b(' + '|'.join(re.escape(s) for s in self.config.normalization.suffixes_to_remove) + r')\b'
            normalized_series = normalized_series.str.replace(suffix_pattern, '')

        # 6. Final cleanup: remove non-alphanumeric, trailing numbers, and extra whitespace
        normalized_series = normalized_series.str.replace(r'[^\w\s]', ' ')
        normalized_series = normalized_series.str.replace(r'\s+\d+$', '')
        normalized_series = normalized_series.str.normalize_spaces()

        # --- Assign the normalized series back to the GPU DataFrame ---
        gdf['normalized_text'] = normalized_series
        
        self.logger.debug(f"Normalization complete. Example: '{gdf[self.config.columns.entity_col].iloc[0]}' -> '{gdf['normalized_text'].iloc[0]}'")
        return gdf
    
    def parse_addresses(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Parses and consolidates addresses, accepting a cuDF DataFrame and managing
        the necessary, temporary data transfers to the CPU for libpostal.

        Args:
            gdf: The input cuDF DataFrame.

        Returns:
            The cuDF DataFrame with parsed and consolidated address information.
        """
        self.logger.info(f"Parsing addresses from: {self.config.columns.address_cols}...")
        
        # --- Step 1: Combine address columns on the GPU ---
        address_cols = self.config.columns.address_cols
        # Start with the first address column
        combined_address_gpu = gdf[address_cols[0]].fillna('').astype(str)
        # Concatenate the rest of the address columns
        for col in address_cols[1:]:
            combined_address_gpu = combined_address_gpu + ' ' + gdf[col].fillna('').astype(str)
        
        # --- Step 2: Move only the combined address to CPU for libpostal ---
        self.logger.info("Moving combined addresses to CPU for parsing...")
        combined_address_cpu = combined_address_gpu.str.normalize_spaces().to_pandas()

        self.logger.info("Expanding and parsing addresses on CPU...")
        # First, expand abbreviations (e.g., St -> Street) - Then, parse the fully expanded address string
        parsed_data = combined_address_cpu.apply(lambda x: utils.safe_parse_address(x, self.logger))

        failures = sum(1 for d in parsed_data if not d)
        if failures > 0:
            self.logger.warning(f"⚠️ Could not parse {failures} addresses.")

        # --- Step 3: Convert parsed data back to a GPU DataFrame ---
        self.logger.info("Moving parsed address data back to GPU...")
        parsed_df = cudf.DataFrame.from_pandas(pd.json_normalize(parsed_data))
        
        # --- Step 4: Merge parsed columns back into the main GPU DataFrame ---
        # Since parsed_df is in the same order as the original gdf, we can
        # reset the index for a clean merge and then restore it.
        original_index = gdf.index
        gdf = gdf.reset_index(drop=True)
        
        addr_cols_map = {
            'addr_street_number': 'address_line_1.street_number',
            'addr_street_name': 'address_line_1.street_name',
            'addr_city': 'city',
            'addr_state': 'state',
            'addr_zip': 'postal_code'
        }
        
        for new_col, old_col in addr_cols_map.items():
            gdf[new_col] = parsed_df[old_col] if old_col in parsed_df.columns else ''
        
        gdf.index = original_index

        # --- Step 5: Create normalized key using the helper ---
        gdf['addr_normalized_key'] = self._create_address_key_gpu(gdf)

        self.logger.info("Consolidating similar addresses on GPU...")
        consolidation_map = self._consolidate_similar_strings_gpu(
            gdf[list(addr_cols_map.keys()) + ['addr_normalized_key']],
            key_col='addr_normalized_key',
            fuzz_ratio=self.config.validation.address_fuzz_ratio
        )
        
        if consolidation_map is not None and len(consolidation_map) > 0:
            gdf['addr_normalized_key'] = gdf['addr_normalized_key'].replace(consolidation_map)
            self.logger.info(f"Consolidated {len(consolidation_map)} address variations on GPU.")

        return gdf
    
    def consolidate_by_address(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        For addresses with multiple different names, this method consolidates them
        to a single canonical name using a GPU-accelerated process.
        """
        self.logger.info("Consolidating names for records sharing an address...")

        # Find how many unique names exist for each address key
        name_counts_per_address = gdf.groupby('addr_normalized_key')['normalized_text'].nunique()
        
        # Identify addresses that have more than one unique name
        addresses_to_consolidate = name_counts_per_address[name_counts_per_address > 1].index
        
        if len(addresses_to_consolidate) == 0:
            self.logger.info("No addresses found with multiple names to consolidate.")
            return gdf

        self.logger.info(f"Found {len(addresses_to_consolidate)} addresses to consolidate names for.")
        
        # Filter the dataframe to only the relevant records to minimize work
        gdf_subset = gdf[gdf['addr_normalized_key'].isin(addresses_to_consolidate)]
        
        # Loop through the addresses that need consolidation
        # This loop is on the CPU, but the work inside is on the GPU.
        address_to_name_map = {}
        for addr_key in addresses_to_consolidate.to_pandas():
            name_series = gdf_subset[gdf_subset['addr_normalized_key'] == addr_key]['normalized_text']
            canonical_name = utils.get_canonical_name_gpu(name_series)
            address_to_name_map[addr_key] = canonical_name
            
        # Create a mapping DataFrame on the GPU
        if not address_to_name_map:
            return gdf

        map_df = cudf.DataFrame({
            'addr_normalized_key': list(address_to_name_map.keys()),
            'canonical_name': list(address_to_name_map.values())
        })
        
        # Merge the map back and update the 'normalized_text' column
        gdf = gdf.merge(map_df, on='addr_normalized_key', how='left')
        gdf['normalized_text'] = gdf['canonical_name'].fillna(gdf['normalized_text'])
        
        return gdf.drop(columns=['canonical_name'])
    
    def encode_entities(self, gdf: cudf.DataFrame, is_training: bool) -> Tuple[cudf.DataFrame, cupy.ndarray]:
        """
        Encodes entity data into numerical vectors using a multi-stream pipeline.

        This method processes each feature type (TF-IDF, phonetic, semantic)
        independently, reduces their dimensionality as needed, and then
        concatenates them into a single, rich feature matrix.

        Args:
            gdf: A cuDF DataFrame containing the pre-processed data.
            is_training: Flag to indicate if the models should be trained.

        Returns:
            A tuple containing:
            - The input cuDF DataFrame.
            - A dense CuPy array of the final, concatenated vectors.
        """
        self.logger.info(f"Encoding entities using multi-stream pipeline: {self.config.modeling.encoders}...")
        sparse_reducers = self.config.modeling.sparse_reducers
        
        # Define the base text for encoders
        base_text_cudf = gdf['normalized_text'].astype(str)
        if self.config.modeling.use_address_in_encoding and 'addr_normalized_key' in gdf.columns:
            base_text_cudf = base_text_cudf + ' ' + gdf['addr_normalized_key'].fillna('').astype(str)
        
        processed_streams = {}

        # --- Stream 1: TF-IDF (Syntactic) ---
        if 'tfidf' in self.config.modeling.encoders:
            self.logger.info("Processing TF-IDF stream...")
            encoder_key = 'tfidf_vectorizer'
            tfidf_params = self.config.modeling.tfidf_params.copy()
            tfidf_params['dtype'] = cupy.float64
            if is_training:
                model = TfidfVectorizer(**tfidf_params)
                sparse_vectors = model.fit_transform(base_text_cudf)
                self.trained_encoders_[encoder_key] = model
            else:
                sparse_vectors = self.trained_encoders_[encoder_key].transform(base_text_cudf)
            
            # Reduce the high-dimensional sparse vectors to dense vectors
            dense_vectors = self._reduce_feature_stream(
                vectors=sparse_vectors,
                stream_name='tfidf',
                svd_params=self.config.modeling.tfidf_svd_params,
                pca_params=self.config.modeling.tfidf_pca_params,
                reducers=sparse_reducers,
                is_training=is_training
            )
            processed_streams['tfidf'] = dense_vectors

        # --- Stream 2: Phonetic ---
        if 'phonetic' in self.config.modeling.encoders:
            self.logger.info("Processing Phonetic stream...")
            encoder_key = 'phonetic_vectorizer'
            
            def multi_phonetic(text): # This function remains on the CPU
                words = text.split()[:self.config.modeling.phonetic_max_words]
                return ' '.join(phonetics.metaphone(w) for w in words if w) or "EMPTY"

            phonetic_text_pd = gdf['normalized_text'].to_pandas().apply(multi_phonetic)
            phonetic_text_cudf = cudf.Series(phonetic_text_pd)
            
            if is_training:
                model = CountVectorizer(**self.config.modeling.phonetic_params)
                sparse_vectors = model.fit_transform(phonetic_text_cudf)
                self.trained_encoders_[encoder_key] = model
            else:
                sparse_vectors = self.trained_encoders_[encoder_key].transform(phonetic_text_cudf)

            # Reduce the sparse vectors to dense vectors
            dense_vectors = self._reduce_feature_stream(
                vectors=sparse_vectors,
                stream_name='phonetic',
                svd_params=self.config.modeling.phonetic_svd_params,
                pca_params=self.config.modeling.phonetic_pca_params,
                reducers=sparse_reducers,
                is_training=is_training
            )
            processed_streams['phonetic'] = dense_vectors

        # --- Stream 3: Semantic ---
        if 'semantic' in self.config.modeling.encoders:
            self.logger.info("Processing Semantic stream...")
            encoder_key = 'semantic_model'
            if is_training:
                model = SentenceTransformer(self.config.modeling.semantic_model)
                self.trained_encoders_[encoder_key] = model
            
            model = self.trained_encoders_.get(encoder_key)
            if model:
                # This stream is already dense and low-dimensional, no reduction needed.
                vectors_cpu = model.encode(
                    gdf['normalized_text'].to_pandas().tolist(), 
                    batch_size=self.config.modeling.semantic_batch_size, 
                    show_progress_bar=self.config.output.log_level <= logging.INFO
                )
                dense_vectors = cupy.asarray(vectors_cpu)
                self.logger.info(f"--> Semantic stream complete. Final shape: {dense_vectors.shape}")
                processed_streams['semantic'] = dense_vectors

        if not processed_streams:
            raise ValueError("No valid encoders specified or processed.")

        # --- Balance the energy of the processed streams ---
        self.logger.info("Balancing energy of feature streams...")
        balanced_vectors_list = utils.balance_feature_streams(
            vector_streams=processed_streams,
            proportions=self.config.modeling.stream_proportions
        )

        # --- Final Step: Concatenate all streams ---
        self.logger.info("Concatenating all feature streams...")
        combined_vectors = cupy.concatenate(balanced_vectors_list, axis=1)
        
        self.logger.info(f"Encoding complete. Final combined vector shape: {combined_vectors.shape}")
        return gdf, combined_vectors
    
    def cluster_entities(self, gdf: cudf.DataFrame, vectors: cupy.ndarray) -> cudf.DataFrame:
        """
        Clusters the low-dimensional vectors using HDBSCAN on the GPU.

        This method fits the HDBSCAN model, stores it for later use in predictions,
        and appends the resulting cluster labels and probabilities to the
        GPU DataFrame.

        Args:
            gdf: The cuDF DataFrame containing the entity data.
            vectors: The final, low-dimensional CuPy array of vectors to be clustered.

        Returns:
            The input cuDF DataFrame with 'cluster' and 'cluster_probability'
            columns added.
        """
        self.logger.info(f"Clustering {vectors.shape[0]} vectors with HDBSCAN...")
        
        # Ensure parameters are copied from the config
        hdbscan_params = self.config.modeling.hdbscan_params.copy()
        
        # Instantiate and fit the GPU-accelerated HDBSCAN model
        clusterer = HDBSCAN(**hdbscan_params)
        clusterer.fit(vectors)
        
        # Store the fitted model to be used in the transform step
        self.cluster_model_ = clusterer
        
        # Assign the labels and probabilities directly to the cuDF DataFrame.
        # No need for .get() as we are staying on the GPU.
        gdf['cluster'] = clusterer.labels_
        gdf['cluster_probability'] = clusterer.probabilities_
        
        # Calculate statistics efficiently on the GPU
        # .nunique() on a filtered series is the correct way to count clusters.
        n_clusters = gdf['cluster'][gdf['cluster'] != -1].nunique()
        n_noise = (gdf['cluster'] == -1).sum()
        
        self.logger.info(f"Clustering found {n_clusters} clusters and {n_noise} noise points.")
        
        return gdf
    
    def cluster_entities_snn(self, vectors: cupy.ndarray) -> cupy.ndarray:
        """
        Generates cluster labels using a three-stage SNN Graph Clustering engine.

        This method provides a set of cluster labels based on graph community
        detection, which can be ensembled with HDBSCAN's output. The process:
        1. Build a mutual-rank weighted k-NN graph to identify strong,
           reciprocal connections.
        2. Find initial communities in the graph using the Louvain algorithm.
        3. Attach noise points to high-confidence clusters based on local
           neighborhood similarity.
        4. Merge over-split clusters based on inter-cluster semantic similarity.

        Args:
            vectors: The final, low-dimensional CuPy array from the UMAP ensemble.

        Returns:
            A CuPy array of the final, processed cluster labels.
        """
        self.logger.info("--- Starting SNN Graph Clustering Engine ---")

        # Ensure vectors are L2-normalized for accurate cosine similarity calculations
        norm = cupy.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / (norm + 1e-8)

        # --- Stage 1: Initial Clustering with Louvain ---
        self.logger.info("Stage 1: Building mutual-rank graph and finding communities...")
        snn_params = self.config.modeling.snn_clustering_params
        
        snn_graph = utils.build_mutual_rank_graph(vectors_norm, k=snn_params['k_neighbors'])
        
        if snn_graph.number_of_edges() == 0:
            self.logger.warning("SNN graph has no edges. All points will be noise.")
            return cupy.full(vectors.shape[0], -1, dtype=cupy.int32)

        # Find communities (partitions) in the graph using the Louvain method
        partitions_df, _ = cugraph.louvain(snn_graph, resolution=snn_params['louvain_resolution'])
        
        # Create a complete labels array for all points, defaulting to noise (-1)
        initial_labels = cupy.full(vectors.shape[0], -1, dtype=cupy.int32)
        
        # Use direct CuPy indexing instead of converting to pandas. This is
        # more efficient as it avoids a GPU -> CPU -> GPU data transfer.
        vertex_indices = partitions_df['vertex'].values
        partition_labels = partitions_df['partition'].values
        initial_labels[vertex_indices] = partition_labels

        n_clusters = len(cupy.unique(initial_labels[initial_labels != -1]))
        n_noise = (initial_labels == -1).sum()
        self.logger.info(f"Louvain found {n_clusters} initial clusters and {n_noise} noise points.")

        # --- Stage 2: Attach Noise Points ---
        self.logger.info("Stage 2: Attaching noise points to confident clusters...")
        noise_params = self.config.modeling.noise_attachment_params
        labels_after_attachment = utils.attach_noise_points(
            vectors=vectors_norm,
            labels=initial_labels,
            k=noise_params['k_neighbors'],
            tau=noise_params['attachment_similarity_threshold'],
            min_matching=noise_params['min_matching_neighbors'],
            ratio_threshold=noise_params['attachment_ratio_threshold']
        )
        
        attached_count = n_noise - (labels_after_attachment == -1).sum()
        self.logger.info(f"Attached {attached_count} noise points.")

        # --- Stage 3: Merge Over-split Clusters ---
        self.logger.info("Stage 3: Merging over-split clusters...")
        final_labels = utils.merge_snn_clusters(
            vectors=vectors_norm,
            labels=labels_after_attachment,
            merge_params=self.config.modeling.cluster_merging_params,
            logger=self.logger
        )

        final_n_clusters = len(cupy.unique(final_labels[final_labels != -1]))
        self.logger.info(f"Cluster merging complete. Final cluster count: {final_n_clusters}")
        self.logger.info("--- SNN Graph Clustering Engine Finished ---")

        return final_labels
    
    def merge_similar_clusters(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Intelligently merges clusters that represent the same real-world entity
        using a GPU-accelerated, graph-based approach.

        Args:
            gdf: The cuDF DataFrame after initial clustering.

        Returns:
            The cuDF DataFrame with cluster labels merged and data harmonized.
        """
        self.logger.info("Merging similar clusters based on name and address...")
        
        # --- Step 1: Create a "profile" for each cluster on the GPU ---
        self.logger.debug("Creating cluster profiles on GPU...")
        unique_clusters = gdf[gdf['cluster'] != -1]['cluster'].unique().to_pandas()
        
        profiles = []
        for cid in unique_clusters:
            cluster_gdf = gdf[gdf['cluster'] == cid]
            # Use GPU-native helpers to find canonical representations
            canonical_name = utils.get_canonical_name_gpu(cluster_gdf['normalized_text'])
            best_addr_row = utils.get_best_address_gpu(cluster_gdf)
            if not best_addr_row.empty:
                canonical_addr = best_addr_row['addr_normalized_key'].iloc[0]
                profiles.append({
                    'cluster': cid, 
                    'name': canonical_name, 
                    'address': canonical_addr, 
                    'size': len(cluster_gdf)
                })
        
        if not profiles:
            self.logger.info("No clusters to merge.")
            return gdf
            
        cluster_profiles_gdf = cudf.DataFrame(profiles)

        # --- Step 2: Merge clusters with EXACT address matches (GPU-native) ---
        self.logger.info("Merging clusters with identical canonical addresses...")
        # Group by address and find addresses shared by multiple clusters
        addr_counts = cluster_profiles_gdf.groupby('address')['cluster'].count()
        exact_match_addresses = addr_counts[addr_counts > 1].index
        
        merge_map = {}
        if len(exact_match_addresses) > 0:
            exact_merge_gdf = cluster_profiles_gdf[cluster_profiles_gdf['address'].isin(exact_match_addresses)]
            # For each address, the winner is the largest cluster
            exact_merge_gdf = exact_merge_gdf.sort_values('size', ascending=False)
            winners = exact_merge_gdf.drop_duplicates(subset=['address'], keep='first')
            
            # Create a map from loser to winner
            losers = exact_merge_gdf[~exact_merge_gdf['cluster'].isin(winners['cluster'])]
            merge_plan = losers.merge(winners[['address', 'cluster']], on='address', suffixes=('_loser', '_winner'))
            
            if not merge_plan.empty:
                exact_map_cpu = merge_plan.to_pandas().set_index('cluster_loser')['cluster_winner'].to_dict()
                merge_map.update(exact_map_cpu)

        # Apply the exact merge map
        if merge_map:
            gdf['cluster'] = gdf['cluster'].replace(merge_map)
            cluster_profiles_gdf['cluster'] = cluster_profiles_gdf['cluster'].replace(merge_map)
            # Consolidate profiles after merge
            cluster_profiles_gdf = cluster_profiles_gdf.drop_duplicates(subset=['cluster'], keep='first')
            self.logger.info(f"Performed {len(merge_map)} exact address merges.")

        # --- Step 3: Use a GRAPH for fuzzy merging (GPU-native) ---
        self.logger.info("Building similarity graph for fuzzy merging...")
        # Define the distance thresholds (1 - similarity ratio)
        name_dist_threshold = 1 - (self.config.validation.name_fuzz_ratio / 100.0)
        addr_dist_threshold = 1 - (self.config.validation.address_fuzz_ratio / 100.0)
        
        # Find similar names using the helper
        name_edges = utils.find_similar_pairs(
            string_series=cluster_profiles_gdf['name'],
            vectorizer_params=self.config.modeling.similarity_tfidf,
            nn_params=self.config.modeling.nearest_neighbors_params,
            similarity_threshold=name_dist_threshold
        )

        # Find similar addresses using the helper
        addr_edges = utils.find_similar_pairs(
            string_series=cluster_profiles_gdf['address'],
            vectorizer_params=self.config.modeling.similarity_tfidf,
            nn_params=self.config.modeling.nearest_neighbors_params,
            similarity_threshold=addr_dist_threshold
        )
        
        # An edge exists in the final graph only if it exists in BOTH name and address graphs
        final_edges = addr_edges.merge(name_edges, on=['source', 'dest'])

        if final_edges.empty:
            self.logger.info("No fuzzy cluster merges found.")
            return gdf

        # Find connected components to identify merge groups
        components = utils.find_graph_components(
            edge_list=final_edges,
            source_col='source',
            dest_col='dest',
            vertex_col_name='profile_idx',
            component_col_name='component_id'
        )
        
        # Map component IDs back to cluster profiles
        profiles_with_components = cluster_profiles_gdf.reset_index().merge(components, left_on='index', right_on='profile_idx')
        
        # --- Step 4: Perform the fuzzy merge ---
        # The winner of each component is the largest original cluster
        profiles_with_components = profiles_with_components.sort_values('size', ascending=False)
        fuzzy_winners = profiles_with_components.drop_duplicates(subset=['component_id'], keep='first')
        
        fuzzy_losers = profiles_with_components[~profiles_with_components['cluster'].isin(fuzzy_winners['cluster'])]
        fuzzy_merge_plan = fuzzy_losers.merge(fuzzy_winners[['component_id', 'cluster']], on='component_id', suffixes=('_loser', '_winner'))
        
        fuzzy_map_cpu = {}
        if not fuzzy_merge_plan.empty:
            fuzzy_map_cpu = fuzzy_merge_plan.to_pandas().set_index('cluster_loser')['cluster_winner'].to_dict()
            merge_map.update(fuzzy_map_cpu)

        # Apply the final combined merge map
        if merge_map:
            self.logger.info(f"Performing {len(fuzzy_map_cpu)} fuzzy merges.")
            gdf['cluster'] = gdf['cluster'].replace(merge_map)
        else:
            self.logger.info("No clusters were merged.")
            return gdf

        # --- Step 5: Harmonize ALL merged clusters ---
        # After merging, ensure all records within a cluster share the same canonical info.
        self.logger.info("Harmonizing data within merged clusters...")
        winner_cluster_ids = cudf.Series(list(set(merge_map.values())))
        
        # Get the new, definitive canonical info for the winning clusters
        definitive_profiles = []
        for cid in winner_cluster_ids.to_pandas():
            cluster_gdf = gdf[gdf['cluster'] == cid]
            definitive_name = utils.get_canonical_name_gpu(cluster_gdf['normalized_text'])
            definitive_addr = cluster_gdf['addr_normalized_key'].value_counts().index[0]
            definitive_profiles.append({
                'cluster': cid,
                'definitive_name': definitive_name,
                'definitive_addr_key': definitive_addr
            })
        
        if definitive_profiles:
            definitive_gdf = cudf.DataFrame(definitive_profiles)
            # Merge the definitive info back into the main dataframe
            gdf = gdf.merge(definitive_gdf, on='cluster', how='left')
            
            # Update records where a definitive profile exists
            update_mask = gdf['definitive_name'].notna()
            gdf.loc[update_mask, 'normalized_text'] = gdf.loc[update_mask, 'definitive_name']
            gdf.loc[update_mask, 'addr_normalized_key'] = gdf.loc[update_mask, 'definitive_addr_key']
            gdf = gdf.drop(columns=['definitive_name', 'definitive_addr_key'])

        self.logger.info("Merge and harmonization complete.")
        return gdf
    
    def consolidate_identical_entities(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Ensures that entities with identical normalized names and addresses are in the same cluster.
        
        Args:
            gdf: The cuDF DataFrame after cluster merging.
            
        Returns:
            The cuDF DataFrame with identical entities consolidated.
        """
        self.logger.info("Consolidating entities with identical names and addresses...")
        
        # Only look at clustered entities
        clustered_gdf = gdf[gdf['cluster'] != -1].copy()
        
        if clustered_gdf.empty:
            return gdf
        
        # Create a composite key of name + address
        clustered_gdf['entity_key'] = (
            clustered_gdf['normalized_text'] + '|||' + 
            clustered_gdf['addr_normalized_key'].fillna('')
        )
        
        # Find entity keys that appear in multiple clusters
        entity_cluster_groups = clustered_gdf.groupby('entity_key')['cluster'].unique()
        multi_cluster_entities = entity_cluster_groups[entity_cluster_groups.list.len() > 1]
        
        if multi_cluster_entities.empty:
            self.logger.info("No identical entities found in different clusters.")
            return gdf
        
        self.logger.warning(f"Found {len(multi_cluster_entities)} entity+address combinations in multiple clusters!")
        
        # Create a consolidation map
        consolidation_map = {}
        
        for entity_key, cluster_array in multi_cluster_entities.to_pandas().items():
            clusters = cluster_array.tolist()
            name, addr = entity_key.split('|||', 1)
            
            # Log the issue
            self.logger.warning(f"  '{name}' at '{addr}' found in clusters: {clusters}")
            
            # Find the best cluster to keep (largest, or highest average probability)
            cluster_stats = []
            for cid in clusters:
                cluster_mask = (gdf['cluster'] == cid)
                size = cluster_mask.sum()
                avg_prob = gdf.loc[cluster_mask, 'cluster_probability'].mean()
                cluster_stats.append({
                    'cluster': cid,
                    'size': size,
                    'avg_prob': avg_prob
                })
            
            # Sort by size (desc), then probability (desc)
            cluster_stats.sort(key=lambda x: (x['size'], x['avg_prob']), reverse=True)
            winner_cluster = cluster_stats[0]['cluster']
            
            # Map all other clusters to the winner
            for cid in clusters:
                if cid != winner_cluster:
                    consolidation_map[cid] = winner_cluster
        
        # Apply the consolidation map
        if consolidation_map:
            self.logger.info(f"Consolidating {len(consolidation_map)} clusters...")
            
            # For efficiency, we'll update clusters in batches
            for old_cluster, new_cluster in consolidation_map.items():
                mask = gdf['cluster'] == old_cluster
                gdf.loc[mask, 'cluster'] = new_cluster
                self.logger.debug(f"  Moved {mask.sum()} records from cluster {old_cluster} to {new_cluster}")
        
        # Clean up the temporary column if we added it
        if 'entity_key' in clustered_gdf.columns:
            gdf = gdf.drop(columns=['entity_key'], errors='ignore')
        
        self.logger.info("Consolidation complete.")
        return gdf
    
    def refine_clusters(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Refines clusters with address enrichment and conflict-based splitting on the GPU.

        Args:
            gdf: The cuDF DataFrame after validation.

        Returns:
            The cuDF DataFrame with clusters refined and data harmonized.
        """
        if 'addr_normalized_key' not in gdf.columns:
            gdf['final_cluster'] = gdf['cluster']
            return gdf

        self.logger.info("Refining clusters with address enrichment and splitting...")
        gdf['final_cluster'] = gdf['cluster']
        gdf['address_was_enriched'] = False
        
        # --- Address Enrichment ---
        self.logger.debug("Performing address enrichment...")
        # First, get the canonical profile for each cluster
        unique_clusters = gdf[gdf['cluster'] != -1]['cluster'].unique().to_pandas()
        canonical_profiles = []
        for cid in unique_clusters:
            cluster_gdf = gdf[gdf['cluster'] == cid]
            best_addr_row = utils.get_best_address_gpu(cluster_gdf)
            if not best_addr_row.empty:
                profile = best_addr_row.iloc[0:1]
                profile['cluster'] = cid
                canonical_profiles.append(profile)

        if not canonical_profiles:
            self.logger.info("No valid clusters to refine.")
            return gdf

        canonical_gdf = cudf.concat(canonical_profiles).rename(columns={
            'addr_street_number': 'c_street_number', 'addr_street_name': 'c_street_name',
            'addr_city': 'c_city', 'addr_state': 'c_state', 'addr_zip': 'c_zip'
        })
        
        # Merge canonical info back to the main DataFrame
        gdf = gdf.merge(canonical_gdf[['cluster', 'c_street_number', 'c_street_name', 'c_city', 'c_state', 'c_zip']], on='cluster', how='left')
        
        # Identify rows that can be enriched
        enrich_mask = (
            (gdf['cluster'] != -1) &
            (gdf['addr_street_name'].isna() | (gdf['addr_street_name'] == '')) &
            (gdf['addr_city'] == gdf['c_city']) &
            (gdf['addr_state'] == gdf['c_state']) &
            (gdf['addr_zip'] == gdf['c_zip'])
        )
        
        enriched_count = enrich_mask.sum()
        if enriched_count > 0:
            self.logger.info(f"Enriching {enriched_count} records with missing street info.")
            gdf.loc[enrich_mask, 'addr_street_number'] = gdf.loc[enrich_mask, 'c_street_number']
            gdf.loc[enrich_mask, 'addr_street_name'] = gdf.loc[enrich_mask, 'c_street_name']
            gdf.loc[enrich_mask, 'address_was_enriched'] = True
        
        gdf = gdf.drop(columns=['c_street_number', 'c_street_name', 'c_city', 'c_state', 'c_zip'])

        # --- Address Conflict Splitting ---
        self.logger.debug("Splitting clusters based on address conflicts...")
        max_cluster_id = gdf['final_cluster'].max()
        
        # Split by State
        if self.config.validation.enforce_state_boundaries:
            state_groups = gdf[gdf['final_cluster'] != -1].groupby('final_cluster')['addr_state'].nunique()
            clusters_to_split_by_state = state_groups[state_groups > 1].index
            if not clusters_to_split_by_state.empty:
                self.logger.info(f"Splitting {len(clusters_to_split_by_state)} clusters with multiple states.")
                state_split_gdf = gdf[gdf['final_cluster'].isin(clusters_to_split_by_state)]
                # Assign a new unique ID to each state within a cluster
                new_ids = state_split_gdf.groupby(['final_cluster', 'addr_state']).ngroup()
                # Use .values to assign the raw array, bypassing the index alignment check.
                gdf.loc[state_split_gdf.index, 'final_cluster'] = max_cluster_id + 1 + new_ids.values
                max_cluster_id = gdf['final_cluster'].max()

        # Split by Street Number Range
        # Convert street numbers to numeric, coercing errors to NaT (which behaves like NaN)
        gdf['addr_street_number_numeric'] = cudf.to_numeric(gdf['addr_street_number'], errors='coerce')
        
        # Group by cluster, street name, and zip to check number ranges
        street_groups = gdf[gdf['final_cluster'] != -1].groupby(['final_cluster', 'addr_street_name', 'addr_zip'])
        street_ranges = street_groups['addr_street_number_numeric'].agg(['min', 'max', 'nunique'])
        
        # Identify groups that have multiple unique numbers and a range exceeding the threshold
        conflicts = street_ranges[
            (street_ranges['nunique'] > 1) &
            ((street_ranges['max'] - street_ranges['min']) > self.config.validation.street_number_threshold)
        ]
        
        if not conflicts.empty:
            self.logger.info(f"Splitting {len(conflicts)} sub-groups with wide street number ranges.")
            # Merge conflict markers back to the main gdf
            gdf = gdf.merge(conflicts.reset_index()[['final_cluster', 'addr_street_name', 'addr_zip']].assign(is_conflict=True),
                            on=['final_cluster', 'addr_street_name', 'addr_zip'], how='left')
            gdf['is_conflict'] = gdf['is_conflict'].fillna(False)

            conflict_gdf = gdf[gdf['is_conflict']]
            # Assign a new unique ID to each street number within a conflict group
            new_ids = conflict_gdf.groupby(['final_cluster', 'addr_street_name', 'addr_zip', 'addr_street_number']).ngroup()
            # Use .values to assign the raw array, bypassing the index alignment check that was causing an error.
            gdf.loc[conflict_gdf.index, 'final_cluster'] = max_cluster_id + 1 + new_ids.values
            gdf = gdf.drop(columns=['is_conflict'])

        # --- Final Cleanup ---
        # Rebuild the normalized key for any records that were enriched
        if enriched_count > 0:
            self.logger.debug("Rebuilding normalized address key for enriched records...")
            gdf.loc[enrich_mask, 'addr_normalized_key'] = utils.create_address_key_gpu(gdf.loc[enrich_mask])

        self.logger.info("Refinement complete.")
        if 'addr_street_number_numeric' in gdf.columns:
            gdf = gdf.drop(columns=['addr_street_number_numeric'])
        return gdf
    
    def build_canonical_map(self, gdf: cudf.DataFrame):
        """
        Builds the final canonical map from the refined clusters on the GPU.

        This method assigns a single canonical name and address profile to each
        cluster and handles "chain" entities (clusters with the same name but
        different addresses) by appending a unique suffix.

        Args:
            gdf: The cuDF DataFrame after all refinement and splitting steps.
        """
        self.logger.info("Building canonical map from refined clusters...")
        
        # --- Step 1: Get Canonical Profile for Each Cluster ---
        unique_clusters = gdf[gdf['final_cluster'] != -1]['final_cluster'].unique().to_pandas()
        
        canonical_profiles = []
        for cid in unique_clusters:
            cluster_gdf = gdf[gdf['final_cluster'] == cid]
            
            # Use our GPU-native helpers to get the best name and address
            canonical_name = utils.get_canonical_name_gpu(cluster_gdf['normalized_text'])
            best_addr_row = utils.get_best_address_gpu(cluster_gdf)

            # Calculate the average probability for this cluster
            avg_prob = cluster_gdf['cluster_probability'].mean()
            
            if not best_addr_row.empty:
                # Create a dictionary from the single-row DataFrame
                profile = best_addr_row.to_pandas().iloc[0].to_dict()
                profile['final_cluster'] = cid
                profile['canonical_name'] = canonical_name
                profile['avg_cluster_prob'] = avg_prob
                canonical_profiles.append(profile)

        if not canonical_profiles:
            self.logger.warning("No valid clusters found to build canonical map.")
            self.canonical_map_ = cudf.DataFrame()
            return

        canonical_map_gdf = cudf.DataFrame(canonical_profiles)
        
        # --- Step 2: Handle "Chain" Entities with Address-Aware Logic ---
        self.logger.info("Identifying and numbering chain entities...")
        
        # First, check if we have any duplicate name+address combinations
        # This would indicate a merging failure that needs to be logged
        name_addr_duplicates = canonical_map_gdf.groupby(['canonical_name', 'addr_normalized_key']).size()
        duplicates_found = name_addr_duplicates[name_addr_duplicates > 1]
        
        if not duplicates_found.empty:
            self.logger.warning(f"Found {len(duplicates_found)} name+address combinations appearing in multiple clusters!")
            # Log details for debugging
            for (name, addr), count in duplicates_found.to_pandas().items():
                self.logger.warning(f"  '{name}' at '{addr}' appears in {count} different clusters")
        
        # Calculate how many UNIQUE ADDRESSES each canonical name has
        name_to_unique_addresses = canonical_map_gdf.groupby('canonical_name')['addr_normalized_key'].nunique()
        
        # Only names with multiple addresses need numbering
        chain_names = name_to_unique_addresses[name_to_unique_addresses > 1].index
        
        if not chain_names.empty:
            self.logger.info(f"Found {len(chain_names)} entity names with multiple addresses")
            
            # Process only entries that need numbering
            chains_gdf = canonical_map_gdf[canonical_map_gdf['canonical_name'].isin(chain_names)].copy()
            non_chains_gdf = canonical_map_gdf[~canonical_map_gdf['canonical_name'].isin(chain_names)]
            
            # Sort by name, then by address (for deterministic ordering), then by cluster
            chains_gdf = chains_gdf.sort_values(['canonical_name', 'addr_normalized_key', 'final_cluster'])
            
            # For each name, number the UNIQUE addresses (not clusters)
            # First, get unique name+address combinations
            unique_name_addr = chains_gdf[['canonical_name', 'addr_normalized_key']].drop_duplicates()
            unique_name_addr = unique_name_addr.sort_values(['canonical_name', 'addr_normalized_key'])
            
            # Assign numbers to unique addresses within each name group
            unique_name_addr['chain_num'] = unique_name_addr.groupby('canonical_name').cumcount() + 1
            
            # Merge the numbers back to all clusters
            chains_gdf = chains_gdf.merge(
                unique_name_addr[['canonical_name', 'addr_normalized_key', 'chain_num']], 
                on=['canonical_name', 'addr_normalized_key'],
                how='left'
            )
            
            # Create the new numbered name
            chains_gdf['canonical_name'] = chains_gdf['canonical_name'] + " - " + chains_gdf['chain_num'].astype(str)
            
            # Log what we're doing for transparency
            chain_summary = chains_gdf.groupby('canonical_name').agg({
                'addr_normalized_key': 'first',
                'final_cluster': 'count'
            })
            self.logger.info("Chain numbering applied:")
            for name, row in chain_summary.head(10).to_pandas().iterrows():
                self.logger.info(f"  '{name}' at '{row['addr_normalized_key']}' ({row['final_cluster']} clusters)")
            
            # Reconstruct the final DataFrame
            canonical_map_gdf = cudf.concat([non_chains_gdf, chains_gdf.drop(columns=['chain_num'])])
        else:
            self.logger.info("No chain entities found (all entity names have unique addresses)")

        # --- Step 3: Final validation check ---
        # Ensure we don't have duplicate canonical_name values in the final map
        final_name_counts = canonical_map_gdf['canonical_name'].value_counts()
        final_duplicates = final_name_counts[final_name_counts > 1]
        
        if not final_duplicates.empty:
            self.logger.error(f"ERROR: Final canonical map has {len(final_duplicates)} duplicate names!")
            for name, count in final_duplicates.head(10).to_pandas().items():
                self.logger.error(f"  '{name}' appears {count} times")
            # This should not happen with the fixed logic
            raise ValueError("Canonical map has duplicate names after chain processing")

        self.canonical_map_ = canonical_map_gdf
        self.logger.info(f"Canonical map with {len(self.canonical_map_)} entries built successfully.")

    def predict_clusters(self, gdf: cudf.DataFrame, vectors: cupy.ndarray) -> cudf.DataFrame:
        """
        Predicts and validates cluster membership for new records.
        """
        if self.cluster_model_ is None or self.canonical_map_ is None:
            raise RuntimeError("The resolver must be fitted before predictions can be made.")
            
        self.logger.info("Predicting clusters for new records...")
        gdf['final_cluster'] = self.cluster_model_.predict(vectors)
        
        prob_map = self.canonical_map_[['final_cluster', 'avg_cluster_prob']]
        gdf = gdf.merge(prob_map, on='final_cluster', how='left').rename(columns={'avg_cluster_prob': 'cluster_probability'})
        gdf['cluster_probability'] = gdf['cluster_probability'].fillna(0.0)
        
        if 'addr_normalized_key' in gdf.columns:
            self.logger.info("Validating new cluster assignments...")
            canonical_profiles = self.canonical_map_[['final_cluster', 'addr_normalized_key', 'addr_state']].rename(columns={
                'addr_normalized_key': 'canonical_addr_key',
                'addr_state': 'canonical_state'
            })
            gdf = self._validate_assignments(gdf, canonical_profiles, cluster_col='final_cluster')
            
        return gdf
    
    def apply_canonical_map(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Applies the final canonical map to the data on the GPU.

        This method merges the canonical names and addresses with the main
        DataFrame and handles any records that were not clustered.

        Args:
            gdf: The cuDF DataFrame containing the refined cluster assignments.

        Returns:
            A cuDF DataFrame with the final canonical entity information appended.
        """
        self.logger.info("Applying canonical map to produce final results...")
        
        # Ensure the canonical map exists before trying to merge
        if self.canonical_map_ is None or self.canonical_map_.empty:
            self.logger.warning("Canonical map is empty. Assigning self-references.")
            gdf['canonical_name'] = gdf['normalized_text']
            gdf['canonical_address'] = gdf['addr_normalized_key']
            # You might want to add other canonical columns here as empty if needed
            return gdf

        map_to_merge = self.canonical_map_[['final_cluster', 'canonical_name', 'addr_normalized_key']].rename(
            columns={'addr_normalized_key': 'canonical_address'}
        )

        # Perform a left merge on the GPU to bring in the canonical information
        gdf_result = gdf.merge(map_to_merge, on='final_cluster', how='left')
        
        # For records that were not clustered (final_cluster == -1), the merge
        # will result in NaNs for the canonical columns. We fill these in.
        # The .fillna() method is an efficient way to handle this.
        gdf_result['canonical_name'] = gdf_result['canonical_name'].fillna(gdf_result['normalized_text'])
        gdf_result['canonical_address'] = gdf_result['canonical_address'].fillna(gdf_result['addr_normalized_key'])
        
        # Apply title case formatting if specified in the config
        if self.config.output.output_format == 'proper':
            gdf_result['canonical_name'] = gdf_result['canonical_name'].str.title()
            
        return gdf_result
    
    def score_confidence(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Calculates comprehensive, nuanced confidence scores on the GPU.

        Args:
            gdf: The cuDF DataFrame after the canonical map has been applied.

        Returns:
            The cuDF DataFrame with 'confidence_score' and 'confidence_category' columns.
        """
        self.logger.info("Scoring confidence of assignments with detailed metrics...")
        
        # --- Step 1: Calculate individual score components ---
        gdf['name_similarity'] = utils.calculate_similarity_gpu(
            gdf['normalized_text'], 
            gdf['canonical_name'].str.lower(),
            self.config.modeling.similarity_tfidf
        )
        gdf['address_confidence'] = utils.calculate_similarity_gpu(
            gdf['addr_normalized_key'], 
            gdf['canonical_address'],
            self.config.modeling.similarity_tfidf
        )
        
        # --- Step 2: Calculate cluster-level metrics ---
        valid_clusters_gdf = gdf[gdf['final_cluster'] >= 0]
        if not valid_clusters_gdf.empty:
            cluster_metrics = valid_clusters_gdf.groupby('final_cluster').agg(
                cluster_size=('normalized_text', 'count'),
                avg_cluster_prob=('cluster_probability', 'mean'),
                name_variation=('name_similarity', 'std')
            ).fillna(0)
            cluster_metrics['cohesion_score'] = (1 - cluster_metrics['name_variation']).clip(0, 1)
            gdf = gdf.merge(cluster_metrics, on='final_cluster', how='left')
        else: # Handle case with no clusters
            gdf['cluster_size'] = 1
            gdf['avg_cluster_prob'] = 0.0
            gdf['cohesion_score'] = 1.0

        # Fill metrics for noise points
        gdf['cluster_size'] = gdf['cluster_size'].fillna(1)
        gdf['cohesion_score'] = gdf['cohesion_score'].fillna(1.0)
        
        # --- Step 3: Calculate final weighted score using vectorized operations ---
        w = self.config.scoring.weights
        
        # Calculate cluster size factor with a logarithmic scale
        log_size = cupy.log1p(gdf['cluster_size'].values) / cupy.log1p(10)
        cluster_size_factor = cudf.Series(log_size, index=gdf.index).clip(upper=1.0)
        
        # Calculate the base score by combining all weighted components
        base_score = (
            gdf['cluster_probability'].fillna(0) * w['cluster_probability'] +
            gdf['name_similarity'] * w['name_similarity'] +
            gdf['address_confidence'] * w['address_confidence'] +
            gdf['cohesion_score'] * w['cohesion_score'] +
            cluster_size_factor * w['cluster_size_factor']
        )
        
        # --- Step 4: Apply penalties ---
        # Penalty for large changes between original and canonical name
        change_magnitude = 1 - self._calculate_similarity_gpu(gdf[self.config.columns.entity_col], gdf['canonical_name'])
        base_score = base_score.where(~(change_magnitude > 0.5), base_score * 0.9)
        
        # Penalty for having been address-enriched (if column exists)
        if 'address_was_enriched' in gdf.columns:
            base_score = base_score.where(~gdf['address_was_enriched'], base_score * 0.95)
            
        # Penalty for small clusters with large name changes
        small_cluster_penalty_mask = (gdf['cluster_size'] <= 2) & (change_magnitude > 0.7)
        base_score = base_score.where(~small_cluster_penalty_mask, base_score * 0.85)

        # --- Step 5: Finalize Score ---
        # For unclustered noise points, use a simpler score
        unclustered_mask = gdf['final_cluster'] == -1
        final_score = base_score.where(~unclustered_mask, gdf['name_similarity'] * 0.5)
        
        gdf['confidence_score'] = final_score.clip(0, 1)
        
        # --- Step 6: Categorize Scores ---
        bins = [0, 0.5, 0.7, 0.85, 1.0]
        labels = ['Low', 'Medium', 'High', 'Very High']
        # cudf.cut is the GPU equivalent of pandas.cut
        gdf['confidence_category'] = cudf.cut(gdf['confidence_score'], bins=bins, labels=labels, include_lowest=True)
        
        self.logger.info("Confidence scoring complete.")
        self.logger.info(f"Average confidence: {gdf['confidence_score'].mean():.3f}")
        return gdf
    
    def flag_for_review(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Flags records that may require manual review based on detailed criteria,
        performing all calculations on the GPU.

        Args:
            gdf: The cuDF DataFrame after confidence scoring.

        Returns:
            The cuDF DataFrame with 'needs_review' and 'review_reason' columns added.
        """
        self.logger.info("Flagging records for manual review...")
        
        if 'confidence_score' not in gdf.columns:
            raise ValueError("Run 'score_confidence' before flagging for review.")

        # --- Step 1: Calculate necessary metrics using vectorized operations ---
        # Calculate the magnitude of change between the original and canonical names.
        change_magnitude = 1 - utils.calculate_similarity_gpu(
            gdf[self.config.columns.entity_col], 
            gdf['canonical_name'],
            self.config.modeling.similarity_tfidf
        )
        
        # --- Step 2: Define all review conditions as boolean masks ---
        # This is far more efficient than iterating or applying a function row-by-row.
        conds = {
            'low_confidence': gdf['confidence_score'] < self.config.output.review_confidence_threshold,
            'drastic_change': change_magnitude > 0.7,
            'single_member_change': (gdf['cluster_size'] == 1) & (change_magnitude > 0.01),
        }
        if 'address_was_enriched' in gdf.columns:
            conds['enriched_low_conf'] = gdf['address_was_enriched'] & (gdf['confidence_score'] < 0.8)

        # --- Step 3: Combine masks to create the final 'needs_review' flag ---
        # Start with a Series of all False.
        needs_review_mask = cudf.Series(False, index=gdf.index)
        # Use the in-place OR operator to efficiently combine all conditions.
        for mask in conds.values():
            needs_review_mask |= mask
            
        gdf['needs_review'] = needs_review_mask
        
        # --- Step 4: Build the 'review_reason' string ---
        # This is a vectorized way to build the reason string without a slow .apply() loop.
        gdf['review_reason'] = ''
        for reason, mask in conds.items():
            # Where the condition mask is True, append the reason string.
            # We add a comma prefix that we'll clean up later.
            gdf['review_reason'] = gdf['review_reason'].where(~mask, gdf['review_reason'] + ',' + reason)
            
        # Remove the leading comma from the populated strings.
        gdf['review_reason'] = gdf['review_reason'].str.lstrip(',')

        review_count = gdf['needs_review'].sum()
        self.logger.info(f"{review_count} records flagged for review.")
        return gdf
    
    def get_review_dataframe(self) -> pd.DataFrame:
        """
        Generates a sorted summary DataFrame for easy review of the resolution results.

        This method is designed to be called after `fit_transform` or `transform`
        has been run. It creates a clean, deduplicated, and sorted pandas
        DataFrame showing the mapping from original entity names and addresses
        to their final canonical forms.

        Returns:
            A pandas DataFrame with four columns, sorted for review:
            - 'original_name'
            - 'original_address'
            - 'canonical_name'
            - 'canonical_address'
        """
        if self.resolved_gdf_ is None or self.resolved_gdf_.empty:
            self.logger.error("Resolver has not been run yet. Call fit_transform() or transform() first.")
            return pd.DataFrame(columns=[
                'original_name', 'original_address', 
                'canonical_name', 'canonical_address'
            ])

        self.logger.info("Generating review DataFrame...")

        # --- Step 1: Create a temporary copy to work with ---
        review_gdf = self.resolved_gdf_.copy()

        # --- Step 2: Reconstruct the original full address ---
        original_address_cols = self.config.columns.address_cols
        review_gdf['original_address'] = review_gdf[original_address_cols[0]].fillna('').astype(str)
        for col in original_address_cols[1:]:
            review_gdf['original_address'] = review_gdf['original_address'] + ' ' + review_gdf[col].fillna('').astype(str)
        review_gdf['original_address'] = review_gdf['original_address'].str.normalize_spaces()

        # --- Step 3: Select, rename, and deduplicate the key columns ---
        original_name_col = self.config.columns.entity_col
        final_review_gdf = review_gdf[[
            original_name_col,
            'original_address',
            'canonical_name',
            'canonical_address'
        ]]
        final_review_gdf = final_review_gdf.rename(columns={original_name_col: 'original_name'})
        unique_mappings = final_review_gdf.drop_duplicates()

        # --- Step 4: Sort the results for easier review ---
        # This groups all original entities under their final canonical form.
        sorted_mappings = unique_mappings.sort_values(
            by=['canonical_name', 'canonical_address', 'original_name']
        )

        # --- Step 5: Convert to Pandas and return ---
        self.logger.info(f"Found {len(sorted_mappings)} unique original -> canonical mappings.")
        return sorted_mappings.to_pandas()
    
    def generate_report(self, original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates an enhanced, final summary report from the resolution results,
        performing all calculations on the GPU.

        Args:
            original_df: The original pandas DataFrame that was input to the pipeline.

        Returns:
            A dictionary containing detailed statistics about the resolution process.
        """
        self.logger.info("Generating enhanced final report...")
        resolved_gdf = self.resolved_gdf_
        
        # --- Calculate Base Statistics ---
        unique_before = original_df[self.config.columns.entity_col].nunique()
        unique_after = resolved_gdf['canonical_name'].nunique()
        original_entity_names_gpu = cudf.from_pandas(original_df[self.config.columns.entity_col])

        # --- Calculate Distributional and Detailed Statistics ---
        # Cluster Size Distribution
        cluster_sizes = resolved_gdf[resolved_gdf['final_cluster'] != -1].groupby('final_cluster').size()
        size_stats = cluster_sizes.describe().to_pandas().to_dict() if not cluster_sizes.empty else {}

        # Confidence Score Distribution
        confidence_stats = resolved_gdf['confidence_score'].describe().to_pandas().to_dict() if 'confidence_score' in resolved_gdf.columns else {}

        # Breakdown of Review Reasons
        review_reasons_breakdown = {}
        if 'needs_review' in resolved_gdf.columns and resolved_gdf['needs_review'].sum() > 0:
            review_gdf = resolved_gdf[resolved_gdf['needs_review']]
            if 'review_reason' in review_gdf.columns and not review_gdf.empty:
                # Move to pandas for robust get_dummies operation
                review_reasons = review_gdf['review_reason'].to_pandas()
                review_reasons_breakdown = review_reasons.str.get_dummies(sep=',').sum().to_dict()

        # --- Assemble the Final Report Dictionary ---
        stats = {
            'summary': {
                'total_records': len(resolved_gdf),
                'unique_entities_before': unique_before,
                'unique_entities_after': unique_after,
                'reduction_rate': 1 - (unique_after / max(unique_before, 1)),
                'records_changed': (original_entity_names_gpu.str.lower() != resolved_gdf['canonical_name'].str.lower()).sum(),
            },
            'clustering_details': {
                'clusters_found': len(self.canonical_map_) if self.canonical_map_ is not None else 0,
                'unclustered_records': (resolved_gdf['final_cluster'] == -1).sum(),
                'chain_entities_found': resolved_gdf['canonical_name'].str.contains(r' - \d+$').sum(),
                'enriched_addresses': resolved_gdf['address_was_enriched'].sum() if 'address_was_enriched' in resolved_gdf.columns else 0,
            },
            'cluster_size_distribution': size_stats,
            'confidence_distribution': confidence_stats,
            'review_summary': {
                'total_records_for_review': resolved_gdf['needs_review'].sum() if 'needs_review' in resolved_gdf.columns else 0,
                'review_reasons_breakdown': review_reasons_breakdown
            }
        }
        
        # Convert all GPU-based scalar values to standard Python numbers for clean output
        for _, content in stats.items():
            if isinstance(content, dict):
                for key, val in content.items():
                    if hasattr(val, 'item'):
                        content[key] = val.item()

        # --- Log the formatted report ---
        self.logger.info("--- Resolution Report ---")
        for key, val in stats['summary'].items():
            val_str = f"{val:.2%}" if 'rate' in key else str(val)
            self.logger.info(f"{key.replace('_', ' ').title():<25}: {val_str}")
        
        self.logger.info("\n--- Clustering Details ---")
        for key, val in stats['clustering_details'].items():
            self.logger.info(f"{key.replace('_', ' ').title():<25}: {val}")

        if stats['cluster_size_distribution']:
            self.logger.info("\n--- Cluster Size Distribution ---")
            dist = stats['cluster_size_distribution']
            self.logger.info(f"{'Mean Size':<25}: {dist.get('mean', 0):.2f}")
            self.logger.info(f"{'Std Dev Size':<25}: {dist.get('std', 0):.2f}")
            self.logger.info(f"{'Min / Max Size':<25}: {int(dist.get('min', 0))} / {int(dist.get('max', 0))}")

        if stats['confidence_distribution']:
            self.logger.info("\n--- Confidence Score Distribution ---")
            dist = stats['confidence_distribution']
            self.logger.info(f"{'Mean Confidence':<25}: {dist.get('mean', 0):.3f}")
            self.logger.info(f"{'Std Dev Confidence':<25}: {dist.get('std', 0):.3f}")
            self.logger.info(f"{'Min / Max Confidence':<25}: {dist.get('min', 0):.3f} / {dist.get('max', 0):.3f}")

        if stats['review_summary']['total_records_for_review'] > 0:
            self.logger.info("\n--- Review Summary ---")
            self.logger.info(f"{'Total For Review':<25}: {stats['review_summary']['total_records_for_review']}")
            for reason, count in stats['review_summary']['review_reasons_breakdown'].items():
                self.logger.info(f"  - {reason.replace('_', ' ').title():<22}: {count}")
                
        return stats
    
    def save_model(self, directory_path: str):
        """
        Saves the fitted resolver to a directory, serializing each component
        using the appropriate method for GPU and CPU objects.

        Args:
            directory_path: The path to the directory where the model will be saved.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Please call fit() first.")
        
        self.logger.info(f"Saving trained resolver to directory: {directory_path}...")
        # Use pathlib.Path for object-oriented path operations
        p = Path(directory_path)
        p.mkdir(parents=True, exist_ok=True)

        # 1. Save the configuration object
        with open(p / 'config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

        # 2. Save the main clustering model
        if self.cluster_model_:
            cuml.common.save(self.cluster_model_, p / 'cluster_model.model')

        # 3. Save the canonical map
        if self.canonical_map_ is not None:
            self.canonical_map_.to_parquet(p / 'canonical_map.parquet')

        # 4. Save the dimensionality reduction models
        for name, model in self.reduction_models_.items():
            path = p / f"{name}.model"
            if isinstance(model, cupy.ndarray):
                # cupy.save appends .npy, so we provide the path without the suffix.
                cupy.save(path.with_suffix(''), model)
            elif name == 'umap_reducer_ensemble' and isinstance(model, list):
                # Save each UMAP model in the ensemble individually.
                for i, umap_model in enumerate(model):
                    ensemble_path = p / f"umap_ensemble_{i}.model"
                    cuml.common.save(umap_model, ensemble_path)
            else:
                cuml.common.save(model, path)

        # 5. Save the trained encoders and scalers
        for name, encoder in self.trained_encoders_.items():
            if hasattr(encoder, 'save'): # Handles SentenceTransformer
                encoder.save(str(p / name))
            else: # For TF-IDF, CountVectorizer, and Scalers
                with open(p / f"{name}.pkl", 'wb') as f:
                    pickle.dump(encoder, f)
                    
        self.logger.info("Model saved successfully.")

    @classmethod
    def load_model(cls, directory_path: str) -> 'EntityResolver':
        """
        Loads a resolver from a directory, reconstructing it from its components.

        Args:
            directory_path: The path to the directory where the model was saved.

        Returns:
            A fully reconstructed and fitted EntityResolver instance.
        """
        print(f"--- Loading trained resolver from {directory_path} ---")
        p = Path(directory_path)

        # 1. Load the configuration to initialize the resolver
        with open(p / 'config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        resolver = cls(config)

        # 2. Load the main clustering model
        cluster_model_path = p / 'cluster_model.model'
        if cluster_model_path.exists():
            resolver.cluster_model_ = cuml.common.load(cluster_model_path)

        # 3. Load the canonical map
        canonical_map_path = p / 'canonical_map.parquet'
        if canonical_map_path.exists():
            resolver.canonical_map_ = cudf.read_parquet(canonical_map_path)

        # 4. Load the dimensionality reduction models
        reduction_model_names = ['variance_filter', 'svd_reducer', 'svd_scaler', 'pca_reducer', 'umap_reducer_ensemble']
        for name in reduction_model_names:
            if name == 'umap_reducer_ensemble':
                ensemble = []
                i = 0
                while True:
                    ensemble_path = p / f"umap_ensemble_{i}.model"
                    if ensemble_path.exists():
                        ensemble.append(cuml.common.load(ensemble_path))
                        i += 1
                    else:
                        break
                if ensemble:
                    resolver.reduction_models_[name] = ensemble
            else:
                path_model = p / f"{name}.model"
                path_npy = p / f"{name}.npy"
                if name == 'variance_filter' and path_npy.exists():
                    resolver.reduction_models_[name] = cupy.load(path_npy)
                elif path_model.exists():
                    resolver.reduction_models_[name] = cuml.common.load(path_model)

        # 5. Load the trained encoders and scalers
        for item in p.iterdir():
            if item.suffix == '.pkl' and 'config' not in item.name:
                encoder_name = item.stem
                with open(item, 'rb') as f:
                    resolver.trained_encoders_[encoder_name] = pickle.load(f)
            elif item.is_dir() and 'semantic_model' in item.name:
                try:
                    from sentence_transformers import SentenceTransformer
                    resolver.trained_encoders_[item.name] = SentenceTransformer(str(item))
                except ImportError:
                    resolver.logger.warning(f"SentenceTransformer not available, skipping {item.name}")

        resolver._is_fitted = True
        print("✓ Model loaded successfully.")
        resolver.logger.info(f"Resolver loaded from {directory_path}. Ready for predictions.")
        return resolver
    
    # =========================================================================
    # == HELPER & UTILITY METHODS
    # =========================================================================

    def _setup_logger(self) -> logging.Logger:
        """Sets up a unique, instance-specific logger."""
        # Use the instance's id to create a unique logger name
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        
        # Prevent log propagation to the root logger
        logger.propagate = False
        
        # Clear existing handlers to avoid duplicate messages
        if logger.hasHandlers():
            logger.handlers.clear()

        # Configure and add a new handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            '%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.config.output.log_level)
        
        return logger
    
    def _prepare_gpu_dataframe(self, df: pd.DataFrame) -> cudf.DataFrame:
        """
        Converts a pandas DataFrame to a cuDF DataFrame for GPU processing.
        This is the designated entry point for data into the GPU pipeline.

        Args:
            df: The input pandas DataFrame.

        Returns:
            A cuDF DataFrame ready for GPU-accelerated processing.
        """
        self.logger.info(f"Moving {len(df)} records from CPU (pandas) to GPU (cudf)...")
        return cudf.from_pandas(df)
    
    def _process_pipeline(self, gdf: cudf.DataFrame, is_training: bool) -> cudf.DataFrame:
        """
        Executes the full, GPU-accelerated entity resolution pipeline.
        
        This method now expects and returns a cuDF DataFrame, managing the
        entire workflow on the GPU.

        Args:
            gdf: The input cuDF DataFrame.
            is_training: A boolean flag to switch between training and prediction paths.

        Returns:
            A cuDF DataFrame with resolved entities and confidence scores.
        """
        # --- Step 1 & 2: Text and Address Processing ---
        self.logger.info("Step 1: Normalizing text data.")
        gdf = self.normalize_text(gdf)

        if self.config.columns.address_cols:
            self.logger.info("Step 2: Parsing and consolidating addresses.")
            # This method encapsulates the necessary, temporary CPU trip for libpostal.
            gdf = self.parse_addresses(gdf)
            if is_training:
                self.logger.info("Step 2a: Consolidating names by address.")
                gdf = self.consolidate_by_address(gdf)

        # --- Step 3 & 4: Vectorization and Dimensionality Reduction ---
        self.logger.info("Step 3: Encoding entities into vectors.")
        gdf, vectors = self.encode_entities(gdf, is_training=is_training)
        
        self.logger.info("Step 4: Reducing vector dimensionality.")
        reduced_vectors = self._run_umap_ensemble(vectors, is_training=is_training)

        # --- Step 5: Branch for Training vs. Prediction ---
        if is_training:
            # --- TRAINING PATH ---
            self.logger.info("Step 5a: Clustering entities with HDBSCAN (Training Path).")
            gdf_hdbscan = self.cluster_entities(gdf.copy(), reduced_vectors)

            # --- Step 5b: Get the SNN engine's high-recall opinion ---
            self.logger.info("Step 5b: Clustering entities with SNN Engine...")
            snn_labels = self.cluster_entities_snn(reduced_vectors)

            # --- Step 5c: Ensemble the two opinions using the advanced method ---
            gdf = self._ensemble_cluster_labels(gdf_hdbscan, snn_labels)
            
            self.logger.info("Step 6a: Validating and merging clusters.")
            gdf = self._validate_cluster_membership_with_reassignment(gdf)

            # CHECK 1: After initial validation (using 'cluster' not 'final_cluster')
            if not self._validate_entities_general(gdf, cluster_col='cluster'):
                self.logger.warning("⚠️ Duplicate entities detected after initial cluster validation")
            
            gdf = self.merge_similar_clusters(gdf)
            # CHECK 2: After merging similar clusters
            if not self._validate_entities_general(gdf, cluster_col='cluster'):
                self.logger.warning("⚠️ Duplicate entities detected after merge_similar_clusters")

            gdf = self.consolidate_identical_entities(gdf)  # Fix duplicate entities!
            # CHECK 3: After consolidation (this should definitely pass!)
            if not self._validate_entities_general(gdf, cluster_col='cluster'):
                self.logger.error("❌ CRITICAL: Duplicates still exist after consolidate_identical_entities!")
                # This is a serious error - consolidation should have fixed all duplicates

            self.logger.info("Step 7a: Refining clusters and building canonical map.")
            gdf = self.refine_clusters(gdf)
            # CHECK 4: After refinement (critical - splitting can reintroduce duplicates!)
            if not self._validate_entities_general(gdf, cluster_col='final_cluster'):
                self.logger.warning("⚠️ Duplicate entities detected after refine_clusters - splitting may have reintroduced them")

            self.build_canonical_map(gdf)
            self.logger.info("Step 8a: Applying canonical map and scoring.")
            gdf_result = self.apply_canonical_map(gdf)

            # CHECK 5: Final check (different validation since canonical names are now applied)
            if not self._validate_canonical_consistency(gdf_result):
                self.logger.error("❌ Canonical names are inconsistent with addresses!")
            
            gdf_result = self.score_confidence(gdf_result)
            gdf_result = self.flag_for_review(gdf_result)
        else:
            # --- PREDICTION (TRANSFORM) PATH ---
            self.logger.info("Step 5b: Predicting clusters for new data (Transform Path).")
            gdf = self.predict_clusters(gdf, reduced_vectors)
            
            self.logger.info("Step 6b: Applying canonical map.")
            gdf_result = self.apply_canonical_map(gdf)
            gdf_result['is_new_entity'] = (gdf_result['final_cluster'] == -1)
        
        self.logger.info("GPU pipeline complete.")
        return gdf_result
    
    def _consolidate_similar_strings_gpu(self, address_gdf: cudf.DataFrame, key_col: str, fuzz_ratio: int) -> Optional[cudf.Series]:
        """
        Finds and consolidates similar address strings using a GPU-accelerated approach,
        selecting the best canonical representation based on completeness and frequency.
        """
        # Get frequency of each unique address key
        freq_map = address_gdf[key_col].value_counts()
        
        # Work with unique addresses to avoid redundant calculations
        unique_addresses = address_gdf.drop_duplicates(subset=[key_col]).set_index(key_col)
        unique_addresses['freq'] = freq_map
        
        if len(unique_addresses) < 2:
            return None

        distance_threshold = 1 - (fuzz_ratio / 100.0)
        self.logger.debug(f"Finding matches in {len(unique_addresses)} unique strings with threshold {distance_threshold:.2f}")

        unique_addresses_series = cudf.Series(unique_addresses.index)
        matched_pairs = utils.find_similar_pairs(
            string_series=unique_addresses_series,
            vectorizer_params=self.config.modeling.similarity_tfidf,
            nn_params=self.config.modeling.nearest_neighbors_params,
            similarity_threshold=distance_threshold
        )

        if matched_pairs.empty:
            self.logger.debug("No fuzzy matches found on GPU.")
            return None

        components = utils.find_graph_components(
            edge_list=matched_pairs,
            source_col='source',
            dest_col='dest',
            vertex_col_name='unique_addr_idx',
            component_col_name='component_id'
        )
        
        # --- Begin Best Representative Selection (GPU Implementation) ---
        # Map component IDs back to the unique addresses DataFrame
        candidates = unique_addresses.reset_index().merge(
            components, left_index=True, right_on='unique_addr_idx'
        )
        
        # Calculate completeness score using vectorized operations
        candidates['score'] = utils.calculate_address_score_gpu(candidates)
        
        # Sort by score, then frequency to find the best candidate in each component
        candidates = candidates.sort_values(['score', 'freq', key_col], ascending=[False, False, True])
        
        # The first entry for each component_id is now the canonical one
        canonical_reps = candidates.drop_duplicates(subset=['component_id'], keep='first')
        
        # Create the final mapping from original key to canonical key
        final_map_df = candidates[[key_col, 'component_id']].merge(
            canonical_reps[['component_id', key_col]],
            on='component_id',
            suffixes=('', '_canonical')
        )
        
        final_map_df = final_map_df[final_map_df[key_col] != final_map_df[f"{key_col}_canonical"]]
        
        if len(final_map_df) == 0:
            return None
            
        return final_map_df.set_index(key_col)[f"{key_col}_canonical"]
    
    def _split_canonical_address(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If configured, splits the 'canonical_address' column into its components.

        Args:
            df: The final pandas DataFrame with a 'canonical_address' column.

        Returns:
            The DataFrame with added address component columns, if enabled.
        """
        if not self.config.output.split_address_components:
            return df
            
        self.logger.info("Splitting canonical address into components...")
        
        if 'canonical_address' not in df.columns:
            self.logger.warning("'canonical_address' column not found, skipping split.")
            return df

        # Apply the existing parsing function to the canonical address
        parsed_data = df['canonical_address'].apply(utils.safe_parse_address)
        
        # Normalize the parsed dictionary data into a DataFrame
        parsed_df = pd.json_normalize(parsed_data)
        
        # Define the mapping from the nested dict structure to final column names
        addr_cols_map = {
            'canonical_street_number': 'address_line_1.street_number',
            'canonical_street_name': 'address_line_1.street_name',
            'canonical_city': 'city',
            'canonical_state': 'state',
            'canonical_zip': 'postal_code'
        }
        
        # Add the new columns to the original DataFrame, handling missing components
        for new_col, old_col in addr_cols_map.items():
            if old_col in parsed_df.columns:
                df[new_col] = parsed_df[old_col]
            else:
                df[new_col] = ''
                
        return df
    
    def _reduce_feature_stream(
        self,
        vectors: cupy.sparse.csr_matrix,
        stream_name: str,
        svd_params: Dict[str, Any],
        pca_params: Dict[str, Any],
        reducers: List[str],
        is_training: bool
    ) -> cupy.ndarray:
        """
        Reduces a high-dimensional sparse feature stream using SVD and PCA.

        This method encapsulates the two-step dimensionality reduction process
        for a single feature type (e.g., TF-IDF or phonetic). It handles
        fitting the models during training and transforming data during prediction.

        Args:
            vectors: The input sparse matrix from an encoder.
            stream_name: A unique name for the stream (e.g., 'tfidf') used for
                         logging and naming the saved models.
            svd_params: A dictionary of parameters for TruncatedSVD.
            pca_params: A dictionary of parameters for PCA.
            reducers: A list of strings specifying which reducers to apply.
            is_training: A flag indicating whether to fit new models.

        Returns:
            A dense CuPy array of the final, low-dimensional vectors for this stream.
        """
        self.logger.info(f"Reducing '{stream_name}' stream from shape {vectors.shape}...")
        # This variable will hold the data as it passes through the pipeline
        current_vectors = vectors

        # --- Step 1: Truncated SVD for initial reduction from sparse matrix ---
        if 'svd' in reducers:
            svd_key = f'{stream_name}_svd_reducer'
            if is_training:
                self.logger.debug(f"Fitting TruncatedSVD for '{stream_name}' to {svd_params['n_components']} components...")
                svd = GPUTruncatedSVD(**svd_params)
                current_vectors = svd.fit_transform(current_vectors)
                self.reduction_models_[svd_key] = svd
            else:
                svd = self.reduction_models_[svd_key]
                current_vectors = svd.transform(current_vectors)

            # This sphericizing/whitening step is probably not a good idea.
            # It scales each component by the inverse of its singular value,
            # ensuring that all components have equal variance and contribute
            # fairly to the downstream PCA and UMAP steps.
            #current_vectors /= (svd.singular_values_ + 1e-8)
            self.logger.debug(f"SVD for '{stream_name}' complete. Shape: {current_vectors.shape}")

        # --- Step 2: PCA for further denoising and reduction ---
        if 'pca' in reducers:
            # Safety check: PCA requires dense input. If SVD was not run,
            # this would be a very memory-intensive operation.
            if not isinstance(current_vectors, cupy.ndarray):
                 self.logger.warning(
                     f"PCA requires a dense array but received sparse. Densifying "
                     f"'{stream_name}' stream. This may cause memory issues."
                 )
                 current_vectors = current_vectors.toarray()
            pca_key = f'{stream_name}_pca_reducer'
            if is_training:
                self.logger.debug(f"Fitting PCA for '{stream_name}' to {pca_params['n_components']} components...")
                pca = PCA(**pca_params)
                current_vectors = pca.fit_transform(current_vectors)
                self.reduction_models_[pca_key] = pca
            else:
                pca = self.reduction_models_[pca_key]
                current_vectors = pca.transform(current_vectors)
        
        final_shape = current_vectors.shape
        self.logger.info(f"--> Reduction for '{stream_name}' complete. Final shape: {final_shape}")
        
        # Ensure the final output is a dense array
        if not isinstance(current_vectors, cupy.ndarray):
            return current_vectors.toarray()
        else:
            return current_vectors
        
    def _run_umap_ensemble(self, combined_vectors: cupy.ndarray, is_training: bool) -> cupy.ndarray:
        """
        Performs manifold learning using a UMAP ensemble with dynamically randomized parameters.

        This method creates a robust final embedding by averaging the results of multiple
        UMAP runs. Each run uses a different, randomly sampled set of hyperparameters
        to explore diverse aspects of the data's underlying manifold structure. This
        approach enhances the stability and reliability of the final entity resolution.

        Key Innovations:
        1.  **Randomized Hyperparameters**: Instead of a fixed progression, parameters like
            n_neighbors, min_dist, spread, and others are randomly sampled from defined
            ranges for each ensemble member, promoting greater diversity.
        2.  **Expanded Parameter Space**: Varies a wider set of UMAP parameters, including
            repulsion_strength, negative_sample_rate, and learning_rate, to cover more
            of the potential solution space.
        3.  **Focused Metric**: Exclusively uses the 'cosine' metric, which is most
            appropriate for normalized, high-dimensional text embeddings.
        4.  **Comprehensive Logging**: Logs the exact, complete set of parameters used for
            each ensemble run, providing full transparency and aiding in debugging.

        Args:
            combined_vectors: A dense CuPy array containing the concatenated
                            outputs from all feature streams.
            is_training: A flag indicating whether to fit new UMAP models (True) or
                        use pre-trained ones for transformation (False).

        Returns:
            A final, dense CuPy array of the low-dimensional embeddings, averaged
            across all ensemble members and ready for clustering.
        """
        self.logger.info(f"Running UMAP ensemble on combined vectors of shape {combined_vectors.shape}...")

        # --- Step 1: L2 Normalize the combined vectors ---
        # This is crucial for UMAP, especially when using cosine-based metrics. It ensures
        # distance is based on the angle (similarity) rather than vector magnitude.
        self.logger.debug("L2-normalizing combined vectors for UMAP...")
        norm = cupy.linalg.norm(combined_vectors, axis=1, keepdims=True)
        # Add a small epsilon to avoid division by zero for any zero-vectors
        vectors_normalized = combined_vectors / (norm + 1e-8)

        # --- Step 2: Define ensemble diversity parameter ranges ---
        # These ranges define the space from which each run's parameters will be randomly sampled.
        n_runs = self.config.modeling.umap_n_runs
        
        param_ranges = self.config.modeling.umap_param_ranges

        # --- Step 3: Run the UMAP Ensemble ---
        umap_key = 'umap_reducer_ensemble'
        self.logger.info(f"Fitting UMAP ensemble with {n_runs} run(s)...")
        
        umap_embeddings = []
        if is_training:
            trained_umaps = []
            ensemble_params_used = []  # Store params for logging/debugging
            
            # Use a dedicated random number generator for reproducibility of the ensemble parameters
            rng = np.random.default_rng(self.config.random_state)

            for i in range(n_runs):
                # Start with a copy of the base parameters
                umap_params = self.config.modeling.umap_params.copy()
                
                # Ensure the metric is always 'cosine' for this application
                umap_params['metric'] = 'cosine'

                # --- Dynamically generate parameters for each ensemble member ---
                if i == 0:
                    # First run uses base parameters as a stable, predictable anchor
                    self.logger.debug(f"UMAP run 1/{n_runs}: Using base parameters as a stable anchor.")
                else:
                    # Subsequent runs use randomly sampled parameters for diversity
                    
                    # 1. Sample core parameters from their defined ranges
                    umap_params['n_neighbors'] = rng.integers(
                        low=param_ranges['n_neighbors'][0], 
                        high=param_ranges['n_neighbors'][1], 
                        endpoint=True
                    )
                    umap_params['min_dist'] = rng.uniform(
                        low=param_ranges['min_dist'][0], 
                        high=param_ranges['min_dist'][1]
                    )
                    umap_params['n_epochs'] = rng.integers(
                        low=param_ranges['n_epochs'][0], 
                        high=param_ranges['n_epochs'][1], 
                        endpoint=True
                    )

                    # 2. Dynamically calculate 'spread' to be > 'min_dist'
                    min_spread = max(umap_params['min_dist'] + 0.1, param_ranges['spread'][0])
                    umap_params['spread'] = rng.uniform(
                        low=min_spread, 
                        high=param_ranges['spread'][1]
                    )

                    # 3. Sample additional optimization parameters
                    umap_params['learning_rate'] = rng.uniform(
                        low=param_ranges['learning_rate'][0], 
                        high=param_ranges['learning_rate'][1]
                    )
                    umap_params['repulsion_strength'] = rng.uniform(
                        low=param_ranges['repulsion_strength'][0], 
                        high=param_ranges['repulsion_strength'][1]
                    )
                    umap_params['negative_sample_rate'] = rng.integers(
                        low=param_ranges['negative_sample_rate'][0], 
                        high=param_ranges['negative_sample_rate'][1], 
                        endpoint=True
                    )

                    # 4. Choose initialization strategy randomly
                    umap_params['init'] = rng.choice(param_ranges['init_strategies'])

                # Set a unique, deterministic random seed for the UMAP algorithm itself
                umap_params['random_state'] = self.config.random_state + i * 104729
                
                # Log the complete set of parameters being used for this run
                param_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in umap_params.items()])
                self.logger.debug(f"UMAP run {i+1}/{n_runs}: {param_str}")
                
                # Fit UMAP with these parameters
                try:
                    umap = UMAP(**umap_params)
                    embeddings = umap.fit_transform(vectors_normalized)
                    
                    # Store results
                    umap_embeddings.append(embeddings)
                    trained_umaps.append(umap)
                    ensemble_params_used.append(umap_params.copy())
                except Exception as e:
                    self.logger.error(f"UMAP run {i+1}/{n_runs} failed with parameters: {param_str}", exc_info=True)
                    continue

            if not trained_umaps:
                self.logger.critical("All UMAP ensemble runs failed. Cannot proceed.")
                raise RuntimeError("UMAP ensemble fitting failed for all members.")
                
            self.reduction_models_[umap_key] = trained_umaps
            self.reduction_models_[f"{umap_key}_params"] = ensemble_params_used
            
        else:
            if umap_key not in self.reduction_models_:
                raise RuntimeError("No pre-trained UMAP models found for prediction.")
            trained_umaps = self.reduction_models_[umap_key]
            ensemble_params_used = self.reduction_models_.get(f"{umap_key}_params", [{}] * len(trained_umaps))
            
            self.logger.info(f"Transforming with {len(trained_umaps)} pre-trained UMAP models...")
            for i, (umap, params) in enumerate(zip(trained_umaps, ensemble_params_used)):
                param_summary = ", ".join([f"{k}={params.get(k, 'N/A')}" for k in ['n_neighbors', 'metric', 'init']])
                self.logger.debug(f"UMAP transform {i+1}/{len(trained_umaps)} ({param_summary})")
                embeddings = umap.transform(vectors_normalized)
                umap_embeddings.append(embeddings)

        if not umap_embeddings:
            self.logger.critical("UMAP ensemble produced no embeddings. Cannot proceed.")
            raise RuntimeError("UMAP ensemble failed to produce any embeddings.")

        # --- Step 4: Combine ensemble results ---
        self.logger.info(f"Averaging embeddings from {len(umap_embeddings)} successful UMAP runs...")
        vectors_umap = self._cosine_consensus_embedding(umap_embeddings)
        
        self.logger.info(f"UMAP ensemble complete. Final vector shape: {vectors_umap.shape}")
        
        # --- Step 5: Optional diagnostic logging ---
        if self.logger.level <= logging.DEBUG and is_training:
            ensemble_std = cupy.std(cupy.stack(umap_embeddings), axis=0).mean()
            self.logger.debug(f"Raw ensemble diversity (mean std. dev. across embeddings before consensus): {ensemble_std:.4f}")
            
            sample_size = min(1000, len(vectors_umap))
            if sample_size > 1:
                sample_idx = cupy.random.choice(len(vectors_umap), sample_size, replace=False)
                sample_dists = cuml.metrics.pairwise_distances(vectors_umap[sample_idx])
                # Check if embeddings are normalized (they should be from kernel PCA)
                norms = cupy.linalg.norm(vectors_umap[sample_idx], axis=1)
                self.logger.debug(
                    f"Final embedding distances (sample of {sample_size}) - "
                    f"Min: {sample_dists.min():.4f}, "
                    f"Mean: {sample_dists.mean():.4f}, "
                    f"Max: {sample_dists.max():.4f}, "
                    f"Norms: Mean={norms.mean():.4f}, Std={norms.std():.4f}"
                )
        
        return vectors_umap
    
    def _cosine_consensus_embedding(
        self,
        embeddings_list: List[cupy.ndarray]
    ) -> cupy.ndarray:
        """
        Builds a consensus embedding using Kernel PCA on the average cosine Gram matrix.

        This method provides a scale and orientation-invariant way to combine embeddings
        from an ensemble. It is mathematically equivalent to performing classical MDS on
        the average angular distances but is formulated more directly as a kernel method.

        Process:
        1. Normalizes each embedding in the list to prepare for cosine similarity.
        2. Samples a subset of points to build an "anchor" model efficiently.
        3. Computes the average Gram (cosine similarity) matrix over the ensemble for the sample.
        4. Performs Kernel PCA on the centered Gram matrix to find the principal components.
        5. Projects the anchor points onto these components to create the anchor embedding.
        6. Projects the remaining out-of-sample points onto the same components for a
            consistent final embedding.

        Args:
            embeddings_list: A list of (n_points, n_dims) CuPy arrays from the UMAP ensemble.
            n_samples: The number of points to use for the initial anchor model.
            n_extension_batch_size: The batch size for processing out-of-sample points.

        Returns:
            A final (n_points, n_dims) consensus embedding, row-normalized.
        """
        n_samples = self.config.modeling.cosine_consensus_n_samples
        n_extension_batch_size = self.config.modeling.cosine_consensus_batch_size
        if not embeddings_list:
            self.logger.error("Cannot create consensus from an empty list of embeddings.")
            raise ValueError("embeddings_list cannot be empty")

        n_points, n_dims = embeddings_list[0].shape
        n_runs = len(embeddings_list)
        self.logger.info(f"Creating consensus from {n_runs} embeddings for {n_points} points.")

        # 1. Normalize all embeddings to make dot product equivalent to cosine similarity
        normalized_embeddings = [utils.normalize_rows(E.astype(cupy.float32)) for E in embeddings_list]

        # 2. Determine sampling strategy
        use_sampling = n_points > n_samples
        if use_sampling:
            self.logger.debug(f"Sampling {n_samples}/{n_points} points to build anchor model.")
            # Ensure reproducible sampling by seeding the random number generator
            cupy.random.seed(int(self.config.random_state))
            sample_indices = cupy.sort(cupy.random.choice(n_points, n_samples, replace=False))
        else:
            self.logger.debug("Using all points to build anchor model (no sampling).")
            sample_indices = cupy.arange(n_points, dtype=cupy.int64)
            n_samples = n_points # Ensure n_samples reflects the actual number used

        # 3. Compute the average Gram matrix on the sample
        self.logger.debug("Calculating average Gram matrix on sample...")
        average_gram_matrix = cupy.zeros((n_samples, n_samples), dtype=cupy.float32)
        for embedding_run in normalized_embeddings:
            # Calculate similarity matrix for this run and add to the running average
            sample_vectors = embedding_run[sample_indices]
            similarity_matrix = sample_vectors @ sample_vectors.T
            average_gram_matrix += similarity_matrix
        average_gram_matrix /= float(n_runs)

        # 4. Perform Kernel PCA on the average Gram matrix
        self.logger.debug("Performing Kernel PCA on the average Gram matrix...")
        
        # Cache centering statistics for out-of-sample extension
        training_kernel_row_means = average_gram_matrix.mean(axis=1, keepdims=True)
        training_kernel_grand_mean = float(average_gram_matrix.mean())

        # Center the kernel
        centered_gram_matrix = utils.center_kernel_matrix(average_gram_matrix)

        # Get the top eigenpairs (principal components) of the centered kernel
        eigenvectors, eigenvalues = utils.get_top_k_positive_eigenpairs(
            symmetric_matrix=centered_gram_matrix, 
            k=n_dims,
            logger=self.logger
        )
        
        # The projection formula requires sqrt(eigenvalue)
        sqrt_eigenvalues = cupy.sqrt(eigenvalues)

        # 5. Create the anchor embedding
        # The coordinates are found by projecting the centered kernel onto the eigenvectors.
        # For the training data, this simplifies to V * sqrt(lambda).
        anchor_embedding = eigenvectors * sqrt_eigenvalues
        normalized_anchor_embedding = utils.normalize_rows(anchor_embedding)

        if not use_sampling:
            self.logger.info("Consensus creation complete (no extension needed).")
            return normalized_anchor_embedding

        # 6. Perform out-of-sample extension for the remaining points
        self.logger.info("Extending consensus embedding to full dataset via kernel projection...")
        final_consensus_embedding = cupy.zeros((n_points, n_dims), dtype=cupy.float32)
        final_consensus_embedding[sample_indices] = normalized_anchor_embedding

        remaining_indices = cupy.setdiff1d(cupy.arange(n_points, dtype=cupy.int64), sample_indices)
        
        # Pre-calculate the inverse for the projection formula to use in the loop
        inverse_sqrt_eigenvalues = 1.0 / (sqrt_eigenvalues + self.config.modeling.epsilon)

        # Pre-slice the sample blocks from each embedding run for efficiency
        sample_blocks = [embedding_run[sample_indices] for embedding_run in normalized_embeddings]

        for start in range(0, remaining_indices.shape[0], n_extension_batch_size):
            stop = min(start + n_extension_batch_size, remaining_indices.shape[0])
            batch_indices = remaining_indices[start:stop]
            
            if start % (10 * n_extension_batch_size) == 0:
                self.logger.debug(f"  Extending points {start}-{stop} of {remaining_indices.shape[0]}...")

            # Calculate the average kernel vector for the batch
            # This is the average similarity of batch points to the anchor points
            avg_kernel_vector = cupy.zeros((batch_indices.shape[0], n_samples), dtype=cupy.float32)
            for embedding_run, sample_block in zip(normalized_embeddings, sample_blocks):
                batch_vectors = embedding_run[batch_indices]
                avg_kernel_vector += batch_vectors @ sample_block.T
            avg_kernel_vector /= float(n_runs)

            # Center the kernel vector using the training statistics
            centered_kernel_vector = utils.center_kernel_vector(
                avg_kernel_vector, 
                training_kernel_row_means, 
                training_kernel_grand_mean
            )

            # Project the centered vector onto the principal components (eigenvectors)
            # Formula: Y_new = K_centered_new @ V * (1/sqrt(lambda))
            batch_consensus_vectors = (centered_kernel_vector @ eigenvectors) * inverse_sqrt_eigenvalues
            
            # Normalize the new vectors and place them in the final output array
            final_consensus_embedding[batch_indices] = utils.normalize_rows(batch_consensus_vectors)

        self.logger.info("Consensus extension complete.")
        return final_consensus_embedding
    
    def _ensemble_cluster_labels(self, gdf_hdbscan: cudf.DataFrame, snn_labels: cupy.ndarray) -> cudf.DataFrame:
        """
        Ensembles clustering results from HDBSCAN and SNN using a purity-based
        mapping and a cluster minting policy.
        """
        params = self.config.modeling.ensemble_params
        self.logger.info("Ensembling HDBSCAN (core) with SNN (rescue) using advanced mapping...")

        # --- Extract arrays from the initial HDBSCAN result ---
        hdb_labels = gdf_hdbscan['cluster'].values
        hdb_probs = gdf_hdbscan['cluster_probability'].values
        n_points = int(hdb_labels.size)

        # --- Build a mapping between SNN and HDBSCAN clusters ---
        # Create a DataFrame to analyze the overlap between the two label sets
        overlap_df = cudf.DataFrame({'hdb': hdb_labels, 'snn': snn_labels})
        
        # We only consider points that were clustered by both algorithms
        clustered_both = overlap_df[(overlap_df['hdb'] != -1) & (overlap_df['snn'] != -1)]
        
        snn_to_hdb_map = cudf.DataFrame({'snn': [], 'hdb': []})
        if not clustered_both.empty:
            # For each (snn_id, hdb_id) pair, count the number of overlapping points
            overlap_counts = clustered_both.groupby(['snn', 'hdb']).size().reset_index(name='overlap_size')
            
            # For each SNN cluster, find its total size within the overlap set
            snn_total_sizes = overlap_counts.groupby('snn')['overlap_size'].sum().reset_index(name='snn_total_size')
            overlap_counts = overlap_counts.merge(snn_total_sizes, on='snn', how='left')
            
            # Purity = (size of overlap with a specific HDB cluster) / (total size of SNN cluster)
            overlap_counts['purity'] = overlap_counts['overlap_size'] / overlap_counts['snn_total_size']

            # For each SNN cluster, find its best HDB match (the one with the largest overlap)
            overlap_counts = overlap_counts.sort_values(['snn', 'overlap_size'], ascending=[True, False])
            best_matches = overlap_counts.drop_duplicates(subset=['snn'], keep='first')

            # Keep only the mappings that meet our purity and overlap thresholds
            snn_to_hdb_map = best_matches[
                (best_matches['purity'] >= params['purity_min']) & 
                (best_matches['overlap_size'] >= params['min_overlap'])
            ][['snn', 'hdb']]

        # --- Start with HDBSCAN's results as the base ---
        final_labels = hdb_labels.copy()
        final_probs = hdb_probs.copy()
        noise_mask = (final_labels == -1)

        # --- Rescue noise points using the high-confidence mapping ---
        # Create a DataFrame of all SNN labels to join with the mapping
        snn_labels_df = cudf.DataFrame({'snn': snn_labels})
        snn_labels_df = snn_labels_df.merge(snn_to_hdb_map, on='snn', how='left')
        mapped_hdb_labels = snn_labels_df['hdb'].fillna(-1).astype('int32').values

        # Find points that were noise in HDBSCAN but belong to an SNN cluster
        # that has a valid mapping to an HDB cluster.
        rescue_mask = noise_mask & (mapped_hdb_labels != -1)
        final_labels[rescue_mask] = mapped_hdb_labels[rescue_mask]

        # --- Mint new clusters for high-quality, SNN-only groups ---
        if params['allow_new_snn_clusters']:
            # Get the global size of every SNN cluster
            snn_global_sizes = cudf.Series(snn_labels).value_counts().reset_index()
            snn_global_sizes.columns = ['snn', 'size']
            
            # Identify SNN clusters that are unmapped, not noise, and large enough
            unmapped_snn = snn_global_sizes.merge(snn_to_hdb_map, on='snn', how='left')
            new_cluster_candidates = unmapped_snn[
                (unmapped_snn['snn'] != -1) & 
                unmapped_snn['hdb'].isnull() & 
                (unmapped_snn['size'] >= params['min_newcluster_size'])
            ]

            if not new_cluster_candidates.empty:
                # Find the next available cluster ID to avoid collisions
                existing_max_id = int(cupy.asnumpy(final_labels).max())
                next_id = existing_max_id + 1
                
                # Create a mapping from the old SNN ID to a new, unique cluster ID
                candidate_ids = new_cluster_candidates['snn'].values
                new_id_map = cudf.DataFrame({
                    'snn': candidate_ids,
                    'new_id': cupy.arange(next_id, next_id + len(candidate_ids), dtype=cupy.int32)
                })
                
                # Assign the new IDs to the relevant points
                snn_labels_df = snn_labels_df.merge(new_id_map, on='snn', how='left')
                new_ids = snn_labels_df['new_id'].fillna(-1).astype('int32').values
                
                # Apply new IDs only to points that were noise in HDBSCAN
                assign_new_mask = noise_mask & (new_ids != -1)
                final_labels[assign_new_mask] = new_ids[assign_new_mask]

        # --- Assign probabilities to all rescued points ---
        rescued_mask = (hdb_labels == -1) & (final_labels != -1)
        if rescued_mask.any():
            final_probs[rescued_mask] = params['default_rescue_conf']

        # 0=hdb_core, 1=snn_rescue_mapped, 2=snn_new_cluster
        label_source = cupy.zeros(n_points, dtype=cupy.int8)
        # All points that were rescued get a source of at least 1
        label_source[rescued_mask] = 1
        # Points that were rescued into a newly minted SNN cluster get source 2
        newly_minted_mask = (hdb_labels == -1) & (mapped_hdb_labels == -1) & (final_labels != -1)
        label_source[newly_minted_mask] = 2

        # --- Write back the final results ---
        gdf_hdbscan['cluster'] = final_labels
        gdf_hdbscan['cluster_probability'] = final_probs
        gdf_hdbscan['label_source'] = cudf.Series(label_source)
        
        # --- Logging ---
        orig_noise = int(noise_mask.sum())
        final_noise = int((final_labels == -1).sum())
        rescued_ct = orig_noise - final_noise
        mapped_ct = int((label_source == 1).sum())
        new_ct = int((label_source == 2).sum())

        self.logger.info(f"Ensemble complete. Rescued {rescued_ct} of {orig_noise} HDBSCAN-noise points.")
        self.logger.info(f"  - via mapped HDB clusters: {mapped_ct}")
        self.logger.info(f"  - via new SNN-only clusters: {new_ct}")
        
        return gdf_hdbscan
    
    def _validate_assignments(self, gdf: cudf.DataFrame, canonical_profiles: cudf.DataFrame, cluster_col: str) -> cudf.DataFrame:
        """
        A unified helper to validate cluster assignments based on state and address similarity.

        Args:
            gdf: The cuDF DataFrame with records to validate.
            canonical_profiles: A cuDF DataFrame with the canonical address and state for each cluster.
            cluster_col: The name of the cluster column to use (e.g., 'cluster' or 'final_cluster').

        Returns:
            The input cuDF DataFrame with invalid members evicted from their clusters.
        """
        gdf_validated = gdf.merge(canonical_profiles, on=cluster_col, how='left')
        
        eviction_mask = cudf.Series(False, index=gdf_validated.index)
        
        # Condition 1: State boundary violations
        if self.config.validation.enforce_state_boundaries:
            state_mismatch_mask = (gdf_validated['addr_state'] != gdf_validated['canonical_state']) & gdf_validated['addr_state'].notna() & gdf_validated['canonical_state'].notna()
            if self.config.validation.allow_neighboring_states:
                allowed_pairs = set(tuple(sorted(p)) for p in self.config.validation.allow_neighboring_states)
                mismatched_pairs = gdf_validated.loc[state_mismatch_mask, ['addr_state', 'canonical_state']].to_pandas()
                mismatched_pairs['sorted_pair'] = mismatched_pairs.apply(lambda r: tuple(sorted(r)), axis=1)
                pairs_to_evict = mismatched_pairs[~mismatched_pairs['sorted_pair'].isin(allowed_pairs)]
                eviction_mask.loc[pairs_to_evict.index] = True
            else:
                eviction_mask |= state_mismatch_mask

        # Condition 2: Address similarity violations
        addr_sim_threshold = self.config.validation.address_fuzz_ratio / 100.0
        compare_mask = gdf_validated['canonical_addr_key'].notna() & gdf_validated['addr_normalized_key'].notna()
        rows_to_compare = gdf_validated[compare_mask]
        
        if not rows_to_compare.empty:
            similarities = utils.calculate_similarity_gpu(
                rows_to_compare['addr_normalized_key'],
                rows_to_compare['canonical_addr_key'],
                self.config.modeling.similarity_tfidf
            )
            failed_mask = (similarities < addr_sim_threshold)
            eviction_mask.loc[rows_to_compare.index[failed_mask]] = True

        # Apply Final Eviction
        final_eviction_mask = eviction_mask & (gdf_validated[cluster_col] != -1)
        evicted_count = final_eviction_mask.sum()
        
        if evicted_count > 0:
            self.logger.info(f"Evicting {evicted_count} records from clusters due to validation failures.")
            gdf_validated.loc[final_eviction_mask, cluster_col] = -1
            if 'cluster_probability' in gdf_validated.columns:
                gdf_validated.loc[final_eviction_mask, 'cluster_probability'] = 0.0
        
        return gdf_validated.drop(columns=['canonical_addr_key', 'canonical_state'])
    
    def _validate_cluster_membership_with_reassignment(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Enhanced validation that not only evicts invalid members but also
        attempts to reassign them to appropriate clusters in one, fully
        GPU-accelerated pass.
        
        This version uses smart batching and filtering to avoid memory explosions
        from cross joins while maintaining GPU acceleration.

        Args:
            gdf: The cuDF DataFrame after initial clustering.
            
        Returns:
            The cuDF DataFrame with validated and potentially reassigned clusters.
        """
        self.logger.info("Validating cluster membership with GPU-accelerated reassignment...")
        
        # --- Step 1: Build Cluster Profiles (fully on GPU) ---
        self.logger.debug("Building cluster profiles on GPU...")
        clustered_gdf = gdf[gdf['cluster'] != -1]
        
        if clustered_gdf.empty:
            self.logger.info("No clusters to validate.")
            return gdf
        
        # Get cluster statistics in one go
        cluster_stats = clustered_gdf.groupby('cluster').agg({
            'cluster_probability': 'mean',
            'normalized_text': 'count'  # This gives us size
        }).reset_index()
        cluster_stats.columns = ['profile_cluster', 'avg_probability', 'size']
        
        # Get canonical names and addresses for each cluster
        unique_clusters = cluster_stats['profile_cluster'].unique()
        
        canonical_profiles_list = []
        for cid in unique_clusters.to_pandas():
            cluster_subset = clustered_gdf[clustered_gdf['cluster'] == cid]
            if cluster_subset.empty: continue
            
            c_name = utils.get_canonical_name_gpu(cluster_subset['normalized_text'])
            best_addr = utils.get_best_address_gpu(cluster_subset)
            
            if not best_addr.empty:
                canonical_profiles_list.append({
                    'profile_cluster': cid,
                    'profile_canonical_name': c_name,
                    'profile_canonical_addr_key': best_addr['addr_normalized_key'].iloc[0],
                    'profile_canonical_state': best_addr['addr_state'].iloc[0]
                })

        if not canonical_profiles_list:
            self.logger.warning("Could not build any valid cluster profiles.")
            return gdf

        profiles_gdf = cudf.DataFrame(canonical_profiles_list)
        profiles_gdf = profiles_gdf.merge(cluster_stats, on='profile_cluster')

        # --- Step 2: First Pass - Validate Current Assignments ---
        self.logger.debug("Validating current cluster assignments...")
        
        # Add current cluster info to each entity
        entities_with_current = gdf.merge(
            profiles_gdf, left_on='cluster', right_on='profile_cluster', how='left'
        )
        
        # Check if entities are valid in their current clusters
        entities_with_current['current_name_sim'] = utils.calculate_similarity_gpu(
            entities_with_current['normalized_text'], 
            entities_with_current['profile_canonical_name'],
            self.config.modeling.similarity_tfidf
        )
        entities_with_current['current_addr_sim'] = utils.calculate_similarity_gpu(
            entities_with_current['addr_normalized_key'], 
            entities_with_current['profile_canonical_addr_key'],
            self.config.modeling.similarity_tfidf
        )
        
        name_threshold = self.config.validation.name_fuzz_ratio / 100.0
        addr_threshold = self.config.validation.address_fuzz_ratio / 100.0
        
        entities_with_current['is_valid_in_current'] = (
            (entities_with_current['current_name_sim'] >= name_threshold) &
            (entities_with_current['current_addr_sim'] >= addr_threshold)
        )
        
        if self.config.validation.enforce_state_boundaries:
            state_ok = self._check_state_compatibility_gpu(
                entities_with_current['addr_state'], entities_with_current['profile_canonical_state']
            )
            entities_with_current['is_valid_in_current'] &= state_ok
        
        # Mark entities that need reassignment (invalid or currently noise)
        entities_with_current['needs_reassignment'] = (
            ~entities_with_current['is_valid_in_current']
        ) | (entities_with_current['cluster'] == -1)
        
        # --- Step 3: Second Pass - Find Better Clusters (only for those that need it) ---
        entities_needing_reassignment = entities_with_current[entities_with_current['needs_reassignment']].copy()
        entities_needing_reassignment['original_index'] = entities_needing_reassignment.index
        
        
        if not entities_needing_reassignment.empty:
            self.logger.debug(f"Finding better clusters for {len(entities_needing_reassignment)} entities...")

            # Drop the profile columns from the entity dataframe before batching
            # to prevent a name collision in the cross-join. These columns were
            # added earlier in this function and are no longer needed.
            profile_cols_to_drop = [
                'profile_cluster', 'profile_canonical_name',
                'profile_canonical_addr_key', 'profile_canonical_state',
                'avg_probability', 'size'
            ]
            # Also drop helper columns created during the initial validation.
            helper_cols_to_drop = [
                'current_name_sim', 'current_addr_sim', 'is_valid_in_current', 'needs_reassignment'
            ]

            # Create a clean DataFrame for batching, using errors='ignore' for safety.
            clean_entities_for_batching = entities_needing_reassignment.drop(
                columns=profile_cols_to_drop + helper_cols_to_drop, errors='ignore'
            )
            
            batch_size = self.config.validation.validate_cluster_batch_size
            best_new_assignments = []
            
            for start_idx in range(0, len(clean_entities_for_batching), batch_size):
                batch = clean_entities_for_batching.iloc[start_idx:start_idx + batch_size]
                batch_best = self._find_best_clusters_batch_gpu(batch, profiles_gdf, name_threshold, addr_threshold)
                best_new_assignments.append(batch_best)
            
            if best_new_assignments:
                all_new_assignments = cudf.concat(best_new_assignments)
                
                # Update the main dataframe with new assignments
                entities_with_current = entities_with_current.merge(
                    all_new_assignments,
                    left_index=True,
                    right_on='original_index',
                    how='left'
                )
        else:
            self.logger.debug("All entities are valid in their current clusters.")

        # --- Step 4: Apply Final Assignments ---
        # For entities that didn't need reassignment, their best cluster is their current one.
        final_cluster = cupy.where(
            entities_with_current['needs_reassignment'].fillna(False).values,
            entities_with_current['best_cluster'].fillna(-1).values,
            entities_with_current['cluster'].fillna(-1).values
        ).astype('int32')

        final_probability = cupy.where(
            entities_with_current['needs_reassignment'].fillna(False).values,
            (entities_with_current['best_score'] * entities_with_current['best_prob']).fillna(0.0).values,
            entities_with_current['cluster_probability'].fillna(0.0).values
        ).astype('float32')

        # Log the changes
        reassigned_mask = (gdf['cluster'] != final_cluster) & (final_cluster != -1) & (gdf['cluster'] != -1)
        evicted_mask = (gdf['cluster'] != -1) & (final_cluster == -1)
        rescued_mask = (gdf['cluster'] == -1) & (final_cluster != -1)
        
        self.logger.info(
            f"Validation results: {int(evicted_mask.sum())} evicted, "
            f"{int(reassigned_mask.sum())} reassigned, "
            f"{int(rescued_mask.sum())} rescued from noise"
        )
        
        # Update the original dataframe
        gdf['cluster'] = final_cluster
        gdf['cluster_probability'] = final_probability
        
        return gdf
    
    def _find_best_clusters_batch_gpu(
        self, 
        entities_batch: cudf.DataFrame, 
        profiles_gdf: cudf.DataFrame,
        name_threshold: float, 
        addr_threshold: float
    ) -> cudf.DataFrame:
        """
        Finds the best cluster for a batch of entities with memory-safe chunking.

        This method avoids GPU out-of-memory errors by breaking the most
        memory-intensive step—the cross-join between entities and cluster
        profiles—into smaller, manageable chunks. It dynamically calculates a
        safe batch size for the profiles and iterates through them, ensuring
        that the temporary DataFrame never exceeds available memory.

        Args:
            entities_batch: A cuDF DataFrame of entities needing cluster assignment.
            profiles_gdf: A cuDF DataFrame of all possible cluster profiles.
            name_threshold: The minimum name similarity for a valid match.
            addr_threshold: The minimum address similarity for a valid match.

        Returns:
            A cuDF DataFrame with the best cluster assignment for each entity.
        """
        n_entities = len(entities_batch)
        n_profiles = len(profiles_gdf)

        # --- Dynamic Batch Size Calculation ---
        # To prevent OOM errors, we calculate a safe number of profiles to
        # process in each chunk. The goal is to keep the number of pairs
        # in the temporary cross-join below a reasonable limit.
        max_pairs_per_chunk = self.config.modeling.profile_comparison_max_pairs_per_chunk
        profile_batch_size = max(1000, max_pairs_per_chunk // n_entities)

        self.logger.debug(
            f"Processing {n_entities} entities against {n_profiles} profiles "
            f"in profile chunks of size {profile_batch_size}"
        )

        all_chunk_results = []

        # --- Iterate Through Profile Chunks ---
        # This loop processes the full set of profiles in smaller batches.
        for profile_start in range(0, n_profiles, profile_batch_size):
            profile_end = min(profile_start + profile_batch_size, n_profiles)
            profile_chunk = profiles_gdf.iloc[profile_start:profile_end]

            # --- Memory-Safe Cross-Join ---
            # The cross-join is performed by adding a dummy key and merging.
            # This is the most memory-intensive operation in the function.
            entities_batch['_dummy'] = 1
            profile_chunk['_dummy'] = 1
            pairs = entities_batch.merge(profile_chunk, on='_dummy', how='outer').drop(columns='_dummy')

            # --- Similarity Calculation and Validation ---
            # These operations are performed on the smaller 'pairs' DataFrame.
            pairs['name_sim'] = utils.calculate_similarity_gpu(
                pairs['normalized_text'], 
                pairs['profile_canonical_name'],
                self.config.modeling.similarity_tfidf
            )
            pairs['addr_sim'] = utils.calculate_similarity_gpu(
                pairs['addr_normalized_key'], 
                pairs['profile_canonical_addr_key'],
                self.config.modeling.similarity_tfidf
            )

            pairs['is_valid'] = (
                (pairs['name_sim'] >= name_threshold) &
                (pairs['addr_sim'] >= addr_threshold)
            )

            if self.config.validation.enforce_state_boundaries:
                state_ok = self._check_state_compatibility_gpu(
                    pairs['addr_state'], pairs['profile_canonical_state']
                )
                pairs['is_valid'] &= state_ok

            valid_pairs = pairs[pairs['is_valid']].copy()

            # Get the scoring weights from the config
            weights = self.config.validation.reassignment_scoring_weights

            if not valid_pairs.empty:
                # --- Scoring and Result Collection ---
                # Calculate a match score for each valid pair.
                size_factor = (cupy.log1p(valid_pairs['size'].values) /
                            cupy.log1p(10)).clip(max=1.0)
                valid_pairs['match_score'] = (
                    weights['name_similarity'] * valid_pairs['name_sim'] +
                    weights['address_similarity'] * valid_pairs['addr_sim'] +
                    weights['cluster_size'] * cudf.Series(size_factor, index=valid_pairs.index) +
                    weights['cluster_probability'] * valid_pairs['avg_probability']
                )

                # Keep only the essential columns to minimize memory footprint.
                chunk_best = valid_pairs[[
                    'original_index', 'profile_cluster',
                    'match_score', 'avg_probability'
                ]].copy()
                all_chunk_results.append(chunk_best)

            # --- Aggressive Memory Cleanup ---
            # Explicitly delete large temporary DataFrames and free the GPU
            # memory pool at the end of each loop to prevent fragmentation.
            del pairs
            del valid_pairs
            if 'chunk_best' in locals():
                del chunk_best
            cupy.get_default_memory_pool().free_all_blocks()

        # --- Final Aggregation ---
        # After checking all chunks, combine the results and find the single
        # best match for each entity across all chunks.
        if not all_chunk_results:
            # Handle the case where no valid matches were found in any chunk.
            result = cudf.DataFrame({'original_index': entities_batch['original_index']})
        else:
            combined = cudf.concat(all_chunk_results, ignore_index=True)
            best_matches = (combined
                            .sort_values('match_score', ascending=False)
                            .drop_duplicates(subset=['original_index'], keep='first'))

            # Merge the best matches back to the original entity batch to ensure
            # all entities are included in the final result.
            result = entities_batch[['original_index']].merge(
                best_matches, on='original_index', how='left'
            )

        # --- Finalize and Return ---
        # Rename columns and fill NaNs for entities that had no valid match.
        result = result.rename(columns={
            'profile_cluster': 'best_cluster',
            'match_score': 'best_score',
            'avg_probability': 'best_prob'
        })

        result['best_cluster'] = result['best_cluster'].fillna(-1).astype('int32')
        result['best_score'] = result['best_score'].fillna(0.0)
        result['best_prob'] = result['best_prob'].fillna(0.0)

        return result
    
    def _check_state_compatibility_gpu(
        self, entity_states: cudf.Series, cluster_states: cudf.Series
    ) -> cudf.Series:
        """Checks state compatibility using GPU operations."""
        # States are compatible if they match, or if one of them is null
        states_match = (entity_states == cluster_states) | entity_states.isna() | cluster_states.isna()
        
        if self.config.validation.allow_neighboring_states:
            # This logic can be complex to vectorize perfectly, but for a small
            # number of allowed pairs, this approach is reasonable.
            mismatched = ~states_match
            if mismatched.any():
                mismatched_df = cudf.DataFrame({
                    's1': entity_states[mismatched],
                    's2': cluster_states[mismatched]
                })
                # Create a canonical pair string for checking
                mismatched_df['pair'] = mismatched_df.apply(
                    lambda row: '|'.join(sorted([row.s1, row.s2])), axis=1
                )
                allowed_pairs = {'|'.join(sorted(p)) for p in self.config.validation.allow_neighboring_states}
                
                is_allowed_neighbor = mismatched_df['pair'].isin(list(allowed_pairs))
                states_match.loc[mismatched.index[is_allowed_neighbor]] = True
                
        return states_match
    
    def _validate_entities_general(self, gdf: cudf.DataFrame, cluster_col: str = 'final_cluster') -> bool:
        """
        Validates that no identical name+address combinations exist in different clusters.
        Works with either 'cluster' or 'final_cluster' column.
        
        Args:
            gdf: The cuDF DataFrame to validate
            cluster_col: The name of the cluster column to check ('cluster' or 'final_cluster')
        
        Returns:
            True if validation passes, False otherwise.
        """
        if cluster_col not in gdf.columns:
            self.logger.error(f"Column '{cluster_col}' not found in DataFrame")
            return False
            
        clustered_gdf = gdf[gdf[cluster_col] != -1].copy()
        
        if clustered_gdf.empty:
            return True
        
        # Create composite key
        entity_keys = (
            clustered_gdf['normalized_text'] + '|||' + 
            clustered_gdf['addr_normalized_key'].fillna('')
        )
        
        # Check for duplicates across clusters
        key_cluster_df = cudf.DataFrame({
            'entity_key': entity_keys,
            'cluster': clustered_gdf[cluster_col]
        })
        
        # Group by entity key and count unique clusters
        unique_clusters_per_key = key_cluster_df.groupby('entity_key')['cluster'].nunique()
        
        # Find any keys that appear in multiple clusters
        duplicates = unique_clusters_per_key[unique_clusters_per_key > 1]
        
        if not duplicates.empty:
            self.logger.error(f"Validation FAILED: {len(duplicates)} entities appear in multiple clusters!")
            # Log some examples
            for entity_key, cluster_count in duplicates.head(5).to_pandas().items():
                name, addr = entity_key.split('|||', 1)
                self.logger.error(f"  '{name}' at '{addr}' appears in {cluster_count} different clusters")
            return False
        
        return True
    
    def _validate_canonical_consistency(self, final_gdf: cudf.DataFrame) -> bool:
        """
        The final and most critical validation check. Ensures that the output
        is logically consistent.

        This method verifies that for any given canonical_name in the final
        output, all records with that name share the exact same canonical_address.
        A failure here indicates a serious flaw in the upstream logic.

        Args:
            final_gdf: The final DataFrame after apply_canonical_map has been run.

        Returns:
            True if the output is consistent, False otherwise.
        """
        self.logger.info("Performing final validation of canonical name/address consistency...")
        
        if 'canonical_name' not in final_gdf.columns or 'canonical_address' not in final_gdf.columns:
            self.logger.warning("Canonical columns not found, skipping canonical consistency check")
            return True

        # We only need to check entities that were successfully clustered.
        # Noise points are expected to be inconsistent.
        clustered_gdf = final_gdf[final_gdf['final_cluster'] != -1]

        # For each canonical_name, count how many unique canonical_addresses it has.
        # In a consistent output, this number should always be 1 for every name.
        # CRITICAL: Use dropna=False to catch cases where some records have addresses and others don't!
        addresses_per_name = clustered_gdf.groupby('canonical_name')['canonical_address'].nunique(dropna=False)

        # Find any names that are linked to more than one address.
        inconsistent_names = addresses_per_name[addresses_per_name > 1]

        if not inconsistent_names.empty:
            self.logger.error(f"❌ CRITICAL: Canonical consistency FAILED! {len(inconsistent_names)} names are linked to multiple addresses!")
            
            # Enhanced debugging: show which clusters are involved
            for name, count in inconsistent_names.head(5).to_pandas().items():
                # Find the specific addresses and clusters for this inconsistent name
                problem_records = clustered_gdf[clustered_gdf['canonical_name'] == name][
                    ['canonical_address', 'final_cluster', 'normalized_text']
                ].drop_duplicates()
                
                self.logger.error(f"\n  '{name}' appears at {count} different addresses:")
                for _, row in problem_records.to_pandas().iterrows():
                    self.logger.error(f"    - Address: '{row['canonical_address']}'")
                    self.logger.error(f"      Cluster: {row['final_cluster']}, Original: '{row['normalized_text']}'")
            
            # Special check for chain numbering failures
            base_names_with_issues = inconsistent_names.index.str.replace(r' - \d+$', '', regex=True).unique()
            self.logger.error(f"\n  Base entity names with numbering issues: {base_names_with_issues.to_pandas().tolist()[:5]}")
            
            # Check specifically for names with mixed null/non-null addresses
            for name in inconsistent_names.index[:5]:
                addr_values = clustered_gdf[clustered_gdf['canonical_name'] == name]['canonical_address']
                has_nulls = addr_values.isnull().any()
                has_values = addr_values.notna().any()
                
                if has_nulls and has_values:
                    self.logger.error(f"\n  WARNING: '{name}' has both populated and missing addresses!")
                    null_count = addr_values.isnull().sum()
                    total_count = len(addr_values)
                    self.logger.error(f"    {null_count}/{total_count} records have missing addresses")
            
            return False

        self.logger.info("✅ Final validation PASSED: Canonical names and addresses are consistent.")
        return True
