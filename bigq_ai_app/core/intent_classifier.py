"""Intent/classification helper using Google AI (Gemini) via google-genai.

This module builds a genai.Client using either a Google AI API key
(GOOGLE_API_KEY) or Vertex AI project/location (GOOGLE_CLOUD_PROJECT,
GOOGLE_CLOUD_LOCATION). It also exposes common generation hyperparameters via
environment variables so you can tune behavior without code changes.

Env vars (all optional unless noted):
  - GOOGLE_API_KEY: If set, uses Google AI API directly.
  - GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION: If set (and no API key),
    uses Vertex AI. Location defaults to 'us-central1' when omitted.
  - LLM_MODEL: Model name (default: 'gemini-2.5-flash').
  - TEMPERATURE: float (default: 0.0)
  - TOP_P: float (default: 0.95)
  - TOP_K: int (default: 40)
  - MAX_TOKENS: int (default: 1024)
  - CANDIDATE_COUNT: int (default: 1)
  - STOP_SEQUENCES: comma-separated strings (default: none)
  - RESPONSE_MIME_TYPE: str (default: 'text/plain')
  - SYSTEM_INSTRUCTION: optional system prompt to steer the model
  - GOOGLE_GENAI_API_VERSION: API version, default 'v1'
"""

from __future__ import annotations

import os
from typing import List, Optional

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from .config import BaseConfig


def _parse_stop_sequences(env_val: Optional[str]) -> Optional[List[str]]:
    if not env_val:
        return None
    seqs = [s.strip() for s in env_val.split(",")]
    return [s for s in seqs if s]


def build_genai_client() -> genai.Client:
    """Create a genai.Client using either GOOGLE_API_KEY or Vertex settings.

    Raises:
        ValueError: if neither API key nor project/location are available.
    """
    # Prefer intent-specific API version, fallback to general
    api_version = BaseConfig.INTENT_API_VERSION
    http_options = HttpOptions(api_version=api_version)

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_options)

    # Fallback to Vertex AI (requires ADC or explicit credentials in env)
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    #location = os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
    # Issue when using GOOGLE_CLOUD_LOCATION=US, Vertex AI requires a specific region 
    # like us-central1 (not just "US"). Otherwise the API call fails with a 404 
    # as it's trying to hit an invalid endpoint.
    location = "us-central1"
    
    vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
    # If vertexai is set to "False" (case-insensitive), disable intent classifier usage

    if project:
        return genai.Client(
            http_options=http_options,
            vertexai=vertexai,
            project=project,
            location=location,
        )

    raise ValueError(
        "Missing credentials. Set GOOGLE_API_KEY for Google AI, or set "
        "GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) to use Vertex AI."
    )


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    candidate_count: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    response_mime_type: Optional[str] = None,
    system_instruction: Optional[str] = None,
) -> str:
    """Generate a response using configured hyperparameters.

    Args:
        prompt: The user prompt/input.
        model: Override model name. If None, uses env LLM_MODEL or default.
        temperature, top_p, top_k, max_tokens, candidate_count,
        stop_sequences, response_mime_type, system_instruction: Optional
        overrides. If None, values are sourced from environment variables.

    Returns:
        The response text from the first candidate.
    """
    client = build_genai_client()

    model_name = (
        model
        or BaseConfig.INTENT_MODEL
        or "gemini-2.5-flash"
    )

    # Resolve hyperparameters: explicit arg overrides config
    cfg = GenerateContentConfig(
        temperature=(temperature if temperature is not None else BaseConfig.INTENT_TEMPERATURE),
        top_p=(top_p if top_p is not None else BaseConfig.INTENT_TOP_P),
        top_k=(top_k if top_k is not None else BaseConfig.INTENT_TOP_K),
        max_output_tokens=(max_tokens if max_tokens is not None else BaseConfig.INTENT_MAX_TOKENS),
        candidate_count=(candidate_count if candidate_count is not None else BaseConfig.INTENT_CANDIDATE_COUNT),
        stop_sequences=(
            _parse_stop_sequences(BaseConfig.INTENT_STOP_SEQUENCES)
            if stop_sequences is None else stop_sequences
        ),
        response_mime_type=(
            response_mime_type if response_mime_type is not None else BaseConfig.INTENT_RESPONSE_MIME_TYPE
        ),
        system_instruction=(system_instruction if system_instruction is not None else BaseConfig.INTENT_SYSTEM_INSTRUCTION),
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg,
    )

    # google-genai returns .text for text/plain; if MIME differs, handle accordingly
    return getattr(resp, "text", str(resp))


def classify_intent(question: str) -> str:
    """Classify a question as 'analytics' or 'non_analytics'.

    Uses the intent system instruction to enforce strict one-token output.
    Any non-matching output is coerced to 'non_analytics' to be safe.
    """
    output = generate(question)
    normalized = (output or "").strip().lower()

    # Heuristic cleanups (in case model returns punctuation or code fences)
    normalized = normalized.strip("`\n\t .:;![](){}\"")

    if normalized == "analytics":
        return "analytics"
    if normalized == "non_analytics":
        return "non_analytics"

    # Some models might return classification with extra text; try contains
    if "analytics" in normalized and "non_analytics" not in normalized:
        return "analytics"
    return "non_analytics"


if __name__ == "__main__":
    # Lightweight smoke test when running as a script.
    # uv run python -m bigq_ai_app.core.intent_classifier
    try:
        out = classify_intent("What is the latest analytics?")
        print("intent:", out)
    except Exception as exc:
        print(f"Generation failed: {exc}")


