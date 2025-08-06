from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Tuple
import logging

@dataclass
class ColumnConfig:
    """Configuration for input data columns."""
    entity_col: str = 'company_name'
    address_cols: List[str] = field(default_factory=lambda: ['address'])

@dataclass
class NormalizationConfig:
    """Configuration for text normalization rules."""
    replacements: Dict[str, str] = field(default_factory=lambda: {
        "traiier": "trailer", "rpr": "repair", "svcs": "service", "svc": "service",
        "ctr": "center", "ctrs": "centers", "cntr": "center", "trk": "truck",
        "auto": "automotive", "auth": "authorized", "dist": "distribution",
        "mfg": "manufacturing", "mfr": "manufacturing", "equip": "equipment",
        "natl": "national", "mgmt": "management", "assoc": "associates"
    })
    suffixes_to_remove: Set[str] = field(default_factory=lambda: {
        "inc", "incorporated", "llc", "ll", "lp", "llp", "ltd", "limited",
        "corp", "corporation", "co", "company", "plc", "pllc",
        "pa", "pc", "sc", "dba", "fka", "aka", "etal", "et al",
        "international", "intl", "usa", "america", "us",
        "group", "grp", "holdings", "ent"
    })

@dataclass
class ModelingConfig:
    """Parameters for the core modeling and vectorization pipeline."""
    encoders: List[str] = field(default_factory=lambda: ['tfidf', 'phonetic', 'semantic'])
    # Note: Do not run sparse_reducers without svd
    sparse_reducers: List[str] = field(default_factory=lambda: ['svd', 'pca'])
    use_address_in_encoding: bool = True
    feature_variance_threshold: float = 0.001
    semantic_batch_size: int = 512
    epsilon: float = 1e-8
    cosine_consensus_n_samples: int = 8192
    cosine_consensus_batch_size: int = 2048
    #semantic_model: str = 'all-MiniLM-L6-v2'   # 384 features
    semantic_model: str = 'all-mpnet-base-v2'   # 768 features
    # To prevent out-of-memory errors on the GPU during the validation
    # and reassignment step, this parameter limits the size of the temporary
    # cross-join DataFrame. Tune this based on available VRAM.
    profile_comparison_max_pairs_per_chunk: int = 1_000_000
    
    hdbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        'min_cluster_size': 3, 
        'min_samples': 1, 
        'cluster_selection_epsilon': 0.15,
        'prediction_data': True,
        'alpha': 0.8,
        'cluster_selection_method': 'leaf',
        'allow_single_cluster': False
    })
    tfidf_params: Dict[str, Any] = field(default_factory=lambda: {
        'analyzer': 'char',
        'ngram_range': (3, 5),
        'max_features': 10000,
        'min_df': 3,
        'sublinear_tf': True,
        'dtype': 'cupy.float64' # quote to avoid cupy import, convert inside method
    })
    tfidf_svd_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_components': 2048, 
        'maxiter': 20,
        'tol': 1e-5
    })
    tfidf_pca_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_components': 768, 
        'svd_solver': 'full',
        'tol': 1e-8,
        'iterated_power': 20
    })
    phonetic_max_words: int = 5
    phonetic_params: Dict[str, Any] = field(default_factory=lambda: {
        'analyzer': 'word',
        'binary': True,
        'max_features': 2000
    })
    phonetic_svd_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_components': 384, 
        'maxiter': 20,
        'tol': 1e-5
    })
    phonetic_pca_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_components': 256, 
        'svd_solver': 'full',
        'tol': 1e-8,
        'iterated_power': 20
    })
    umap_n_runs: int = 3    # How many members in the ensemble
    umap_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_neighbors': 15,
        'n_components': 48,
        'min_dist': 0.05,
        'spread': 0.5,
        'metric': 'cosine',
        'init': 'spectral',
        'n_epochs': 400,  
        'negative_sample_rate': 7,  
        'repulsion_strength': 1.0,
        'learning_rate': 0.5
    })
    nearest_neighbors_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_neighbors': 20,
        'metric': 'cosine'
    })
    similarity_tfidf: Dict[str, Any] = field(default_factory=lambda: {
        'analyzer': 'char',
        'ngram_range': (3, 5),
        'min_df': 2,
        'sublinear_tf': True,
        'norm': 'l2'
    })
    # --- Feature Stream Proportions ---
    # Defines the desired final "energy" or variance contribution of each stream.
    # Must sum to 1.0.
    stream_proportions: Dict[str, float] = field(default_factory=lambda: {
        'semantic': 0.45,
        'tfidf': 0.40,
        'phonetic': 0.15
    })
    # --- SNN Graph Clustering Engine Parameters ---
    snn_clustering_params: Dict[str, Any] = field(default_factory=lambda: {
        # k for the initial k-Nearest Neighbors graph. A larger k makes the
        # graph more connected and robust to noise, but increases compute time.
        'k_neighbors': 40,
        # Louvain resolution parameter. Higher values (>1.0) lead to more,
        # smaller communities. Lower values (<1.0) lead to fewer, larger ones.
        'louvain_resolution': 0.65,
    })

    noise_attachment_params: Dict[str, Any] = field(default_factory=lambda: {
        # k for the noise attachment kNN. This should be smaller than the
        # main k to focus on the immediate local neighborhood.
        'k_neighbors': 15,
        # The minimum mean similarity a noise point must have to the neighbors
        # of a candidate cluster to be attached.
        'attachment_similarity_threshold': 0.82,
        # The minimum number of neighbors within the candidate cluster that
        # must meet the similarity threshold.
        'min_matching_neighbors': 2,
        # The similarity to the best cluster must be this much greater
        # than the similarity to the second-best cluster. Prevents ambiguous pulls.
        'attachment_ratio_threshold': 1.5,
    })

    cluster_merging_params: Dict[str, Any] = field(default_factory=lambda: {
        # The median similarity between two clusters must exceed this value
        # for them to be considered for merging.
        'merge_median_threshold': 0.84,
        # The maximum similarity (i.e., the single most similar pair of points)
        # between two clusters must also exceed this value.
        'merge_max_threshold': 0.90,
        # To avoid O(n^2) comparisons, we sample points from large clusters
        # when checking for merges.
        'merge_sample_size': 20,
        # To manage memory when pre-filtering, process the cluster centroids
        # in batches of this size.
        'merge_batch_size': 1000,
        # In the pre-filtering step, only cluster pairs whose centroids have a
        # similarity above this threshold will be considered for a detailed check.
        'centroid_similarity_threshold': 0.75,
        # To speed up centroid calculation for very large clusters,
        # a random sample of this size will be used.
        'centroid_sample_size': 1000
    })
    # --- Advanced Ensemble Clustering Parameters ---
    ensemble_params: Dict[str, Any] = field(default_factory=lambda: {
        # The minimum fraction of an SNN cluster's members that must share the
        # same HDBSCAN cluster ID for a valid mapping to be created.
        'purity_min': 0.75,
        # The minimum number of overlapping members between an SNN and HDBSCAN
        # cluster for a valid mapping to be considered.
        'min_overlap': 2,
        # If True, allows the creation of new clusters from SNN groups that
        # HDBSCAN entirely missed.
        'allow_new_snn_clusters': True,
        # SNN-only clusters must be at least this large to be minted as new.
        'min_newcluster_size': 4,
        # The default confidence score assigned to points that are "rescued"
        # from HDBSCAN's noise by the SNN engine.
        'default_rescue_conf': 0.60,
    })
    umap_param_ranges: Dict[str, Any] = field(default_factory=lambda: {
        'n_neighbors': (5, 65),          # int: from very local to more global structure
        'min_dist': (0.0, 0.15),          # float: from tight packing to more spread out
        'spread': (0.3, 1.5),             # float: controls the scale of the embedding
        'n_epochs': (200, 500),           # int: from faster/coarser to slower/finer optimization
        'learning_rate': (0.3, 2.0),      # float: step size for optimization
        'repulsion_strength': (0.3, 1.5), # float: how much to push non-neighbor points apart
        'negative_sample_rate': (3, 30),  # int: number of negative samples per positive one
        'init_strategies': ['spectral', 'random'] # list: diverse starting points
    })

@dataclass
class ValidationConfig:
    """Parameters for validating and refining matched entities."""
    street_number_threshold: int = 50
    address_fuzz_ratio: int = 87
    name_fuzz_ratio: int = 89
    enforce_state_boundaries: bool = True
    # Defaulting to an empty list enforces identical states.
    # Populate this only if you need to allow specific cross-border matching.
    allow_neighboring_states: List[Tuple[str, str]] = field(default_factory=list)
    validate_cluster_batch_size: int = 1000

    # Weights for scoring potential matches during the reassignment step.
    # This controls which cluster an unassigned point will be moved to.
    # Must sum to 1.0.
    reassignment_scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'name_similarity': 0.40,
        'address_similarity': 0.40,
        'cluster_size': 0.10,
        'cluster_probability': 0.10
    })

@dataclass
class ConfidenceScoringConfig:
    """Weights for calculating the final confidence score."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'cluster_probability': 0.25,
        'name_similarity': 0.20,
        'address_confidence': 0.25,
        'cohesion_score': 0.15,
        'cluster_size_factor': 0.15
    })

@dataclass
class OutputConfig:
    """Configuration for the output and review process."""
    output_format: str = 'proper'
    review_confidence_threshold: float = 0.75
    log_level: int = logging.INFO
    split_address_components: bool = False

@dataclass
class ResolverConfig:
    """
    Master configuration object for the EntityResolver.
    
    This object composes all subordinate configurations for different
    stages of the entity resolution process.
    """
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    scoring: ConfidenceScoringConfig = field(default_factory=ConfidenceScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_state: int = 42

    def __post_init__(self):
        """Ensure random_state is propagated where needed."""
        self.modeling.umap_params['random_state'] = self.random_state
        self.modeling.tfidf_pca_params['random_state'] = self.random_state
        self.modeling.phonetic_pca_params['random_state'] = self.random_state
