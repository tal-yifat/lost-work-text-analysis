import logging
import numpy as np
from typing import List, Dict, Optional
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

// optional: from joblib import load  # if you want to load pickled objects

class PredictionExplainer:
    """
    Explains specific predictions after a pipeline and LLM have generated cluster-level explanations.
    Assumes you have access to a pipeline object (possibly unpickled) that contains:
      - pca_model
      - kmeans_model
      - cluster_data
      - llm_response (from ClusterExplainer)
      - (optionally) the trained HF model for new embeddings
    """

    def __init__(self, pipeline, llm_response):
        """
        :param pipeline: A trained ClusterExplanationPipeline object 
                         (with fitted PCA, KMeans, cluster_data, etc.)
        :param llm_response: The pydantic-validated LLM response from ClusterExplainer
        """
        self.pipeline = pipeline
        self.llm_response = llm_response

    def explain_data_points(self, tokenized_dataset_indices: List[int]):
        """
        For each index, compute new embedding & cluster label, then print a short explanation 
        using cluster rationale from llm_response.
        """
        if self.llm_response is None or not self.llm_response.clusters:
            logger.error("No LLM response or no clusters. Cannot provide explanations.")
            return

        for idx in tokenized_dataset_indices:
            // Your real logic to build embedding for a single data point
            # example:
            # embedding = ...
            embedding = np.random.randn(self.pipeline.pca_components or 768)
            # apply optional PCA
            if self.pipeline.use_pca and self.pipeline.pca_model is not None:
                embedding = self.pipeline.pca_model.transform([embedding])[0]
            # get cluster label
            cluster_id = self.pipeline.kmeans_model.predict([embedding])[0]

            # fetch cluster explanation data
            cluster_llm_obj = self.llm_response.clusters[cluster_id]
            # A typical real scenario: retrieve the original text, the final probability, etc.
            narrative_sample = f"Placeholder text for index {idx}"

            print("\n--- Explanation for data idx", idx, "---")
            print("Narrative:", narrative_sample)
            print("Assigned cluster:", cluster_id)
            print(f"Cluster Name: {cluster_llm_obj.cluster_name}")
            print("Cluster Rationale:", cluster_llm_obj.cluster_rationale)
            print("Tailored Explanation Instructions:", 
                  cluster_llm_obj.cluster_tailored_explanation_instructions)

    def demo_predictions(self):
        """
        Sort the clusters by mean predicted probability, pick a random sample from each,
        and display a high-level explanation.
        """
        if not self.pipeline.cluster_data or not self.llm_response:
            logger.warning("No pipeline cluster_data or LLM response to demo.")
            return

        # Sort cluster_data by mean_pred_proba 
        sorted_clusters = sorted(self.pipeline.cluster_data, key=lambda c: c["mean_pred_proba"])
        for cinfo in sorted_clusters:
            cid = cinfo["cluster_id"]
            mean_prob = cinfo["mean_pred_proba"]
            cluster_llm_obj = self.llm_response.clusters[cid]

            # pick random sample from random_samples
            sample_text = random.choice(cinfo["random_samples"]) if cinfo["random_samples"] else ""

            print(f"\n=== Cluster {cid} ===")
            print(f"Mean Probability: {mean_prob:.4f}")
            print(f"Name: {cluster_llm_obj.cluster_name}")
            print(f"Rationale:\n{cluster_llm_obj.cluster_rationale}")
            print("\nA random sample narrative from this cluster:\n", sample_text) 