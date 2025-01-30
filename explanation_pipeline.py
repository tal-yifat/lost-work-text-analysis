import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ClusterExplanationPipeline:
    """
    Pipeline to produce cluster explanations. Provides:
      - model & dataset loading
      - optional PCA
      - embeddings + predicted prob weighting (based on target variance proportion)
      - clustering
      - cluster data preparation
      - display methods

    The pipeline combines PCA-transformed embeddings with predicted probabilities.
    To control the influence of probabilities in the final space:
    - User specifies target_prop (e.g., 0.1)
    - Pipeline computes alpha so predicted probabilities contribute target_prop
      of total variance in combined space
    - Formula: alpha = sqrt((target_prop * sum_pc_var) / (var_pp * (1 - target_prop)))
      where sum_pc_var = sum of PCA variances, var_pp = variance of predicted probs
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        data_collator: DataCollator,
        openai_api_key: str,
        text_column: str = "text",  
        use_pca: bool = True,
        pca_components: int = 50,
        target_prop: float = 0.1,  
        min_n_clusters: int = 4, 
        max_n_clusters: int = 12,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the pipeline.
        
        Args:
            model: HuggingFace model for embeddings and predictions
            tokenizer: Associated tokenizer
            data_collator: For batching tokenized inputs
            openai_api_key: For LLM-based explanations
            text_column: Name of column containing text data
            use_pca: Whether to apply PCA to embeddings
            pca_components: Number of PCA components if use_pca is True
            target_prop: Target proportion of variance from predicted probability
            min/max_n_clusters: Range for number of clusters
            device: Optional device override (default: auto-detect)
            random_seed: For reproducibility
        """
        if not 0 < target_prop < 1:
            raise ValueError("target_prop must be between 0 and 1 (exclusive)")

        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.openai_api_key = openai_api_key
        self.text_column = text_column
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.target_prop = target_prop
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed

        self.pca_model: Optional[PCA] = None
        self.kmeans_model: Optional[KMeans] = None
        self.cluster_data: Optional[List[Dict]] = None
        self.mean_probs: Optional[np.ndarray] = None
        self.alpha: Optional[float] = None
        self.llm_response: Optional[BaseModel] = None
        self.texts: Optional[List[str]] = None  

        # set seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def compute_alpha(self, pca_embeddings: np.ndarray, probs: np.ndarray) -> float:
        """
        Compute alpha to scale predicted probabilities. When alpha * prob is appended
        to PCA embeddings, it will contribute target_prop of total variance.

        Formula derivation:
        - Let V_p = variance of predicted probs
        - Let V_e = sum of PCA embedding variances
        - We want: (alpha²V_p)/(V_e + alpha²V_p) = target_prop
        - Solving for alpha: alpha = sqrt((target_prop * V_e)/(V_p * (1-target_prop)))
        """
        if not self.use_pca or self.pca_model is None:
            raise ValueError("PCA must be used and fitted before computing alpha")
            
        pca_variances = self.pca_model.explained_variance_
        pred_prob_variance = np.var(probs)
        
        if pred_prob_variance <= 0:
            raise ValueError("Predicted probability variance must be > 0")
            
        logger.debug(
            f"Computing alpha with: target_prop={self.target_prop}, "
            f"sum_pca_var={np.sum(pca_variances):.4f}, pred_var={pred_prob_variance:.4f}"
        )
        
        numerator = self.target_prop * np.sum(pca_variances)
        denominator = pred_prob_variance * (1 - self.target_prop)
        alpha = np.sqrt(numerator / denominator)
        
        logger.info(f"Computed alpha={alpha:.4f} for target_prop={self.target_prop}")
        return alpha

    def build_embeddings_and_probs(
        self, 
        tokenized_dataset, 
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each sample in dataset, extracts:
        - The final hidden state (768-dim) from the [CLS] token
        - The predicted probability (model's classification head)
        Returns tuple of (embeddings, probs) arrays
        """
        logger.info("Building embeddings and predicted probabilities...")
        self.model.eval().to(self.device)

        # Create dataloader directly from the tokenized tensors
        dataloader = DataLoader(
            [{k: v[i:i+1] for k, v in tokenized_dataset.items()} 
            for i in range(len(tokenized_dataset['input_ids']))],
            batch_size=batch_size,
            collate_fn=self.data_collator
        )

        all_embeddings = []
        all_probs = []

        with torch.no_grad():
            total_batches = len(dataloader)
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx}/{total_batches}")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, output_hidden_states=True)
                
                last_hidden_state = outputs.hidden_states[-1]
                cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                
                all_embeddings.append(cls_embeddings)
                all_probs.append(probs)

        embeddings = np.concatenate(all_embeddings)
        probs = np.concatenate(all_probs)
        
        # Store probs as class attribute
        self.probs = probs

        logger.info(f"Built embeddings shape: {embeddings.shape}, probabilities shape: {probs.shape}")
        return embeddings, probs


    def optional_pca_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply PCA if `use_pca` is True. Fit PCA if not fitted, else just transform.
        """
        if not self.use_pca:
            return embeddings

        if self.pca_model is None:
            logger.info(f"Fitting PCA with n_components={self.pca_components}")
            self.pca_model = PCA(n_components=self.pca_components, random_state=self.random_seed)
            return self.pca_model.fit_transform(embeddings)
        else:
            logger.info("PCA model already fitted. Transforming new data.")
            return self.pca_model.transform(embeddings)

    def find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Find optimal k in [min_n_clusters, max_n_clusters] using silhouette scores.
        Falls back to elbow method if sklearn not available or for small datasets.
        """
        logger.info(f"Finding optimal k in range [{self.min_n_clusters}, {self.max_n_clusters}]")
        
        # For very large datasets, sample to speed up silhouette computation
        MAX_SILHOUETTE_SAMPLES = 10000
        if embeddings.shape[0] > MAX_SILHOUETTE_SAMPLES:
            indices = np.random.choice(
                embeddings.shape[0], 
                MAX_SILHOUETTE_SAMPLES, 
                replace=False
            )
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings

        best_score = -1
        best_k = self.min_n_clusters
        k_values = range(self.min_n_clusters, self.max_n_clusters + 1)
        
        for k in k_values:
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.random_seed,
                n_init="auto"
            )
            labels = kmeans.fit_predict(sample_embeddings)
            
            # Compute silhouette score
            score = silhouette_score(sample_embeddings, labels)
            logger.debug(f"k={k}, silhouette_score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logger.info(f"Selected optimal k={best_k} (silhouette_score={best_score:.3f})")
        return best_k

    def perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run KMeans with optimal k if not fitted. Return cluster assignments.
        """
        if self.kmeans_model is None:
            # Find optimal k and create new model
            n_clusters = self.find_optimal_k(embeddings)
            logger.info(f"Running KMeans with optimal n_clusters={n_clusters}")
            self.kmeans_model = KMeans(
                n_clusters=n_clusters, 
                random_state=self.random_seed,
                n_init="auto"
            )
            return self.kmeans_model.fit_predict(embeddings)
        else:
            logger.info("KMeans model already fitted. Predicting clusters for new data.")
            return self.kmeans_model.predict(embeddings)

    def tokenize_data(self, dataset) -> Dict:
        """
        Tokenize input texts using the pipeline's tokenizer.
        Stores original texts for later use.
        """
        logger.info("Tokenizing input data...")
        
        # Store original texts
        self.texts = dataset[self.text_column].tolist()
        
        # Tokenize
        tokenized = self.tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return tokenized

    def prepare_cluster_data(
        self, 
        embeddings: np.ndarray, 
        probs: np.ndarray, 
        labels: np.ndarray
    ) -> List[Dict[str, Union[int, float, List[str]]]]:
        """
        For each cluster, compute mean prob, pick examples near centroid, etc.
        Uses self.texts to access original narratives.
        """
        if self.texts is None:
            raise ValueError("No texts available. Did you run tokenization?")
        
        logger.info("Preparing cluster data for LLM explanation...")
        cluster_data = []
        
        n_clusters = len(np.unique(labels))
        for c_id in range(n_clusters):
            cluster_mask = (labels == c_id)
            cluster_probs = probs[cluster_mask]
            mean_pred = float(cluster_probs.mean()) if len(cluster_probs) > 0 else 0.0

            # Get indices of points closest to centroid
            centroid = self.kmeans_model.cluster_centers_[c_id]
            cluster_points = embeddings[cluster_mask]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_indices = np.argsort(distances)[:5]
            
            # Get random indices
            all_indices = np.where(cluster_mask)[0]
            random_indices = np.random.choice(
                all_indices, 
                size=min(10, len(all_indices)), 
                replace=False
            )

            # Get actual text samples using stored texts
            cluster_indices = np.where(cluster_mask)[0]
            top_centroid_samples = [self.texts[cluster_indices[i]] for i in closest_indices]
            random_samples = [self.texts[i] for i in random_indices]

            cluster_dict = {
                "cluster_id": c_id,
                "mean_pred_proba": mean_pred,
                "top_centroid_samples": top_centroid_samples,
                "random_samples": random_samples,
            }
            cluster_data.append(cluster_dict)

        # Add size statistics
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        logger.info("Cluster sizes: %s", cluster_sizes)

        return cluster_data

    def display_cluster_explanations(self, cluster_data: Optional[List[Dict]] = None):
        """
        Previously in ClusterExplainer. Display cluster name, rationale, etc.
        """
        if cluster_data is None:
            cluster_data = self.cluster_data
        if not cluster_data:
            logger.warning("No cluster_data found. Cannot display explanations.")
            return
        if not self.llm_response:
            logger.warning("LLM response is None. Nothing to display.")
            return

        # If mismatch in lengths, log a warning
        if len(self.llm_response.clusters) != len(cluster_data):
            logger.warning(
                "Number of clusters in LLM response does not match cluster_data. Displaying best effort."
            )

        for idx, cluster_obj in enumerate(self.llm_response.clusters):
            cinfo = cluster_data[idx]
            cid = cinfo["cluster_id"]
            mean_prob = cinfo["mean_pred_proba"]
            print(f"\n=== Cluster {cid} ===")
            print(f"Name: {cluster_obj.cluster_name}")
            print(f"Mean Probability: {mean_prob:.4f}")
            print(f"Rationale:\n{cluster_obj.cluster_rationale}")

    def deep_dive_single_cluster(self, cluster_idx: int):
        """
        Previously in ClusterExplainer. Show cluster data and LLM output.
        """
        if self.cluster_data is None or cluster_idx >= len(self.cluster_data):
            print(f"Invalid cluster_idx {cluster_idx} or no cluster_data.")
            return
        cinfo = self.cluster_data[cluster_idx]
        cluster_llm_obj = self.llm_response.clusters[cluster_idx] if self.llm_response else None

        print(f"\n=== Deep Dive: Cluster {cinfo['cluster_id']} ===")
        print(f"Mean Probability: {cinfo['mean_pred_proba']:.4f}")
        print("Top 5 near centroid:")
        for s in cinfo["top_centroid_samples"]:
            print("- ", s)
        print("\n10 random picks:")
        for s in cinfo["random_samples"]:
            print("- ", s)

        if cluster_llm_obj:
            print("\nLLM Output:")
            print(f" Name: {cluster_llm_obj.cluster_name}")
            print(f" Rationale: {cluster_llm_obj.cluster_rationale}")
            print(" Tailored Explanation Instructions:")
            print(cluster_llm_obj.cluster_tailored_explanation_instructions)

    def compute_probability_quantiles(self, probs: np.ndarray) -> str:
        """
        Compute quantile description of predicted probabilities distribution.
        Returns a string like "10% of data points have p < 0.1, 80% have 0.1 <= p < 0.4..."
        """
        logger.info("Computing probability distribution quantiles...")
        
        # Calculate quantiles at 20% intervals
        quantiles = np.quantile(probs, [0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Format the description
        ranges = []
        prev_q = 0
        for i, q in enumerate(quantiles):
            percent = 20
            if i == 0:
                ranges.append(f"{percent}% of data points have p < {q:.2f}")
            else:
                ranges.append(f"{percent}% have {prev_q:.2f} <= p < {q:.2f}")
            prev_q = q
        
        return ", ".join(ranges)

    def generate_cluster_explanations(self, model_name: str = "gpt-4o", temperature: float = 0.3) -> None:
        """
        Generate explanations for clusters using ClusterExplainer.
        Requires cluster_data to be populated.
        Stores LLM response in self.llm_response.
        """
        if not self.cluster_data:
            raise ValueError("Must run clustering before generating explanations")
        
        from cluster_explanation import ClusterExplainer
        
        # Compute probability distribution description
        prob_quantiles = self.compute_probability_quantiles(self.probs)
        
        logger.info("Generating cluster explanations using LLM...")
        explainer = ClusterExplainer(
            num_clusters=len(self.cluster_data),
            cluster_data=self.cluster_data,
            predicted_probability_quantiles=prob_quantiles,
            model_name=model_name,  
            temperature=temperature
        )
        
        # Get LLM response and store it
        self.llm_response = explainer.get_response()
        if not self.llm_response:
            logger.error("Failed to get LLM response")
            return
        
        logger.info("Successfully generated cluster explanations")

    def run_pipeline(self, dataset) -> List[Dict]:
        """
        Main orchestrator. 
        1) Tokenize input data
        2) Build embeddings & probs
        3) Optional PCA
        4) Compute alpha and combine embeddings with weighted prob
        5) Fit or predict clusters
        6) Prepare cluster data
        7) Generate cluster explanations
        8) Display explanations
        """
        # Tokenize first
        tokenized_dataset = self.tokenize_data(dataset)
        
        embeddings, probs = self.build_embeddings_and_probs(tokenized_dataset)
        
        # Store all probs for later quantile computation
        self.probs = probs
        
        # First apply PCA to get variances for alpha computation
        if self.use_pca:
            embeddings = self.optional_pca_transform(embeddings)
            # Now compute alpha based on PCA variances and prob distribution
            self.alpha = self.compute_alpha(embeddings, probs)
            # Add weighted probability column
            probs_column = (self.alpha * probs).reshape(-1, 1)
            embeddings = np.hstack([embeddings, probs_column])
        
        labels = self.perform_clustering(embeddings)
        
        # Store cluster_data as an attribute
        self.cluster_data = self.prepare_cluster_data(embeddings, probs, labels)
        
        # Generate explanations
        self.generate_cluster_explanations()
        
        # Display the explanations
        logger.info("Displaying cluster explanations...")
        self.display_cluster_explanations()
        
        return self.cluster_data 