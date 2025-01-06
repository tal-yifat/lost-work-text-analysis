"""
Module for clustering explanations using an LLM, with a dedicated ClusterExplainer class.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import openai
from pydantic import BaseModel, Field
from openai import RateLimitError, APIConnectionError, APIError
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
        messages: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4",
         openai_api_key: str = "",
    ):
        self.messages = messages if messages is not None else []
        self.model = model
        if openai_api_key:
            openai.api_key = openai_api_key

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
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                **kwargs
            )
            logger.debug(f"OpenAI API raw response: {response}")

            choice = response.choices[0]
            message = choice.message
            if message.get("content"):
                logger.info("Received valid completion from OpenAI API.")
                return {"content": message.content}
            elif message.get("function_call"):
                return {"function_call": message.function_call}
            else:
                logger.error("Unexpected response structure from OpenAI API.")
                raise OpenAIAPIError("Unexpected response structure from OpenAI.")
        except (RateLimitError, APIConnectionError, APIError) as e:
            logger.error(f"OpenAI error: {str(e)}")
            raise OpenAIAPIError(f"OpenAI error: {e}") from e

    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Appends the user prompt to message history, calls the API, returns the textual content.
        """
        self.messages.append({"role": "user", "content": prompt})

        api_kwargs = {"temperature": temperature, **kwargs}

        try:
            completion_dict = self._call_openai_api_structured_output(**api_kwargs)
            if not completion_dict:
                logger.error("Empty completion from OpenAI API.")
                return None
            content = completion_dict.get("content")
            if content:
                # Store the assistant's message
                self.messages.append({"role": "assistant", "content": content})
                return content
            else:
                logger.error("No content in completion.")
                return None
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
        self.prompt_template = (
            "You are an advanced AI assistant specialized in explaining risk predictions. "
            "Below is information about a set of clusters derived from text narratives "
            "of mine injuries, each cluster representing similar 'rationale' patterns "
            "leading to predictions.\n\n"
            "We have {num_clusters} clusters total. "
            "For each cluster, you must provide:\n"
            "1) cluster_i_name: A short descriptive name.\n"
            "2) cluster_i_rationale: A general rationale or explanation for the risk patterns in that cluster.\n"
            "3) cluster_i_tailored_explanation_instructions: Guidance on how to generate a short, customized explanation "
            "for a specific data point belonging to this cluster (based on that point's narrative and predicted probability).\n\n"
            "**Your response must be valid JSON** that matches the following structure:\n"
            "{\n"
            "  \"clusters\": [\n"
            "    {\n"
            "      \"cluster_name\": \"...\",\n"
            "      \"cluster_rationale\": \"...\",\n"
            "      \"cluster_tailored_explanation_instructions\": \"...\"\n"
            "    },\n"
            "    ... one object per cluster ...\n"
            "  ]\n"
            "}\n\n"
            "Do not output any extra keys or text. The audience are business stakeholders handling claims. "
            "They need a clear, business-oriented rationale for why these injuries might lead to short or long absences. \n"
            "Predicted Probability Quantiles:\n{predicted_probability_quantiles}\n"
        )

    def build_clustering_prompt(self) -> str:
        """
        Builds the final prompt to feed to the LLM, combining the template
        with cluster details.
        """
        # Format the header with base info
        prompt_header = self.prompt_template.format(
            num_clusters=self.num_clusters,
            predicted_probability_quantiles=self.predicted_probability_quantiles
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
        raw_content = self.msg_manager.get_completion(
            prompt=prompt, temperature=self.temperature
        )

        if not raw_content:
            logger.error("LLM returned no content.")
            return None

        # Attempt to parse as JSON into our schema
        try:
            as_dict = json.loads(raw_content)
            validated_obj = self.ResponseSchema(**as_dict)
            return validated_obj
        except (json.JSONDecodeError, pydantic.ValidationError) as exc:
            logger.error(f"Failed to parse LLM JSON response: {exc}")
            return None