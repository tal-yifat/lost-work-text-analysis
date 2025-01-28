"""
Module for clustering explanations using an LLM, with a dedicated ClusterExplainer class.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import openai
import pydantic
from pydantic import BaseModel, Field
from openai import RateLimitError, APIConnectionError, APIError
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

# ----------------------------
# Setup logger
# ----------------------------
def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger that writes to a file with the current date and also outputs to console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_dir, f"openai_calls_{current_date}.log")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger(__name__)

# ----------------------------
# Custom Exceptions & Retry
# ----------------------------
class OpenAIAPIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass

retry_decorator = retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)

# ----------------------------
# Common MessageManager
# ----------------------------
class MessageManager:
    """
    A chat interface with OpenAI's API, including retries, specialized in structured output.
    """
    def __init__(
        self,
        openai_api_key: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o",
    ):
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
           raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=openai_api_key)
        self.messages = messages if messages is not None else []
        self.model = model
        
    @retry_decorator
    def _call_openai_api_structured_output(self, **kwargs) -> Dict[str, Any]:
        """
        Calls the OpenAI API with the stored messages, expecting structured output.
        """
        try:
            logger.info(
                f"Calling OpenAI API with model: {self.model}, "
                f"messages count: {len(self.messages)}, kwargs: {kwargs}"
            )
            completion = self.client.beta.chat.completions.parse(model=self.model, messages=self.messages, **kwargs)
            logger.debug(f"Response from OpenAI API: {completion}")
            response = completion.choices[0].message
            if response.parsed:
                logger.debug(
                    "\nthe response was parsed successfully. Data type: {type(response.parsed)}; \nresponse: {response.parsed}"
                )
                return response.parsed
            elif response.refusal:
                logger.error(f"LLM Refusal: {response.refusal} for request: {kwargs}")
                raise OpenAIAPIError(f"LLM Refusal: {response.refusal}")
        except (RateLimitError, APIConnectionError, APIError) as e:
            logger.error(f"OpenAI error: {str(e)}")
            raise OpenAIAPIError(f"OpenAI error: {e}") from e

    def get_completion(
        self,
        prompt: str,
        response_format: BaseModel,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Appends the user prompt to message history, calls the API, returns the textual content.
        """
        self.messages.append({"role": "user", "content": prompt})

        api_kwargs = {"temperature": temperature, **kwargs}
        if response_format:
            api_kwargs["response_format"] = response_format
        else:
            raise ValueError("response_format is required.")

        try:
            completion = self._call_openai_api_structured_output(**api_kwargs)
            if completion is None:
                logger.error("Received None completion from OpenAI API.")
                return None
            self.messages.append({"role": "assistant", "content": completion})
            return completion
        except RetryError as e:
            logger.error(f"Max retries exceeded: {e}")
            raise OpenAIAPIError("Max retries exceeded calling OpenAI.") from e
        except OpenAIAPIError as e:
            logger.error(f"OpenAI API Error: {e}")
            raise

# ----------------------------
# The ClusterExplainer Class
# ----------------------------
class ClusterExplainer:
    """
    Encapsulates the logic for prompting an LLM to explain a set of clusters,
    returning a structured response.
    """

    class ResponseSchema(BaseModel):
        """
        The structured response schema from the LLM.
        Similar to the older ClusteringLLMResult but nested within the class.
        """
        class ClusterResult(BaseModel):
            cluster_name: str = Field(description="A short descriptive name for the cluster")
            cluster_rationale: str = Field(description="A general rationale or explanation for the risk patterns in that cluster")
            cluster_tailored_explanation_instructions: str = Field(description="Guidance on how to generate a short, customized explanation for a specific data point belonging to this cluster (based on that point's narrative and predicted probability)")

        clusters: List[ClusterResult] = Field(description="A list of clusters, each with a name, rationale, and tailored explanation instructions")

    def __init__(
        self,
        num_clusters: int,
        cluster_data: List[Dict[str, Any]],
        predicted_probability_quantiles: str,
        model_name: str = "o1",
        openai_api_key: str = "",
        temperature: float = 0.3,
    ):
        """
        :param num_clusters: Number of total clusters (e.g., 6).
        :param cluster_data: List of cluster dictionaries, each containing:
               { "cluster_id", "mean_pred_proba", "top_centroid_samples", "random_samples" }.
        :param predicted_probability_quantiles: A string describing how many data points fall into each probability bin.
        :param model_name: The LLM model to use (default "o1").
        :param openai_api_key: The API key if needed.
        :param temperature: LLM temperature.
        """
        self.num_clusters = num_clusters
        self.cluster_data = cluster_data
        self.predicted_probability_quantiles = predicted_probability_quantiles
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.temperature = temperature

        # The message manager handles the actual OpenAI calls
        self.msg_manager = MessageManager(
            model=self.model_name,
            openai_api_key=self.openai_api_key
        )

        # Store or build a base prompt template
        self.prompt_template = '''You are an advanced AI assistant specialized in explaining risk predictions generated by machine learning models. \
Your task is to provide explanations for the predicted probabilities outputted by a transformer-based NLP model. \
The model receives narratives of mine injuries and predicts the probabilities that the injured workers will miss at least 90 days of work. \
The predictions have been classified into a set of {num_clusters} clusters based on their predicted probabilities and embedding representation of the injury narratives. \
Your task is to identify the key patterns in the data for each cluster and provide a general explanation for the predicted level of risk of data points in that cluster.

For each cluster, you should provide:
1) `cluster_i_name`: A short descriptive name.
2) `cluster_i_rationale`: A general rationale or explanation for the risk patterns in that cluster. \
The explanation should not explicitly refer to "cluster", but rather use terms such as "this type of injury", so it is accessible to non-technical audience.
3) `cluster_i_tailored_explanation_instructions`: Guidance on how to generate a short, customized explanation for a specific data point belonging to this cluster \
(based on that point's narrative and predicted probability). Keep in mind that the customized explanation should avoid unsubstantiated guesses about what drove \
the model to make the prediction, and instead focus on interpreting the prediction in light of the patterns in the data for the respective cluster.

The audience are business stakeholders handling disability insurance claims, who are seeking to identify early on insurance claims at risk of turning into cases \
of long-term disability. This could support early intervention that would help workers return to work sooner. The stakeholders need a clear, business-oriented \
rationale for why these injuries might lead to short or long absences. Note that to be effective, the names and rationales for each cluster should not only \
represent that data points in each cluster, but should also clearly differentiate the clusters from each other.

As context, here is information about the distribution of the predicted probabilities for the entire dataset:
{predicted_probability_quantiles}

Below is information about each of the clusters, including:
- The mean predicted probability
- 6 prototypical examples, closest to the cluster's centroid
- 10 random example, representing the variation within the cluster.
'''

    def build_clustering_prompt(self) -> str:
        """
        Builds the final prompt to feed to the LLM, combining the template
        with cluster details.
        """
        # Format the header with base info
        prompt_header = self.prompt_template.format(
            num_clusters=self.num_clusters,
            predicted_probability_quantiles=self.predicted_probability_quantiles,
        )

        # Build cluster details
        prompt_clusters = []
        for cinfo in self.cluster_data:
            cid = cinfo["cluster_id"]
            mean_prob = cinfo["mean_pred_proba"]
            top_samples = cinfo["top_centroid_samples"]
            random_samples = cinfo["random_samples"]

            cluster_text = (
                f"\n\nCluster {cid} - mean predicted probability: {mean_prob}\n\n"
                f"Cluster {cid} - Top 5 near centroid:\n{top_samples}\n\n"
                f"Cluster {cid} - 10 random picks:\n{random_samples}\n\n"
            )
            prompt_clusters.append(cluster_text)

        final_prompt = prompt_header + "\n".join(prompt_clusters)
        return final_prompt

    def get_response(self) -> Optional[ResponseSchema]:
        """
        Calls the LLM with the constructed prompt, parses the JSON,
        and returns a pydantic-validated ResponseSchema object.
        """
        prompt = self.build_clustering_prompt()
        response = self.msg_manager.get_completion(
            prompt=prompt, 
            temperature=self.temperature,
            response_format=self.ResponseSchema
        )

        if not response:
            logger.error("LLM returned no content.")
            return None
        
        return response