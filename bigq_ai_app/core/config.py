import os
from dotenv import load_dotenv

load_dotenv()

class BaseConfig:
    APP_NAME = "BigQ AI Chatbot"
    DEBUG = False
    LOG_LEVEL = "INFO"

    # GCP
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "US")
    BQ_CONNECTION = os.getenv("BQ_CONNECTION_NAME", "bq-llm-connection") 
    BQ_DATASET = os.getenv("BQ_DATASET", "bq_llm")

    # GCS for multimodal/vision (optional)
    GCS_BUCKET = os.getenv("GCS_BUCKET", "")
    VISION_OUTPUT_PREFIX = os.getenv("VISION_OUTPUT_PREFIX", "vision_ocr_output/")

    # Models
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro") # Latest Gemini model as of 2025-09-11
    # Max tokens for response. Recommended values: 1024, 2048, 5120
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "5120")) 
    # Temperatue minimizes randomness to produce the most likely, syntactically correct SQL.
    # 0.0 (more deterministic) to 1.0 (more creative). Recommended values: 0.0, 0.2
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0")) 
    # Nucleus Sampling (topP) narrows the token selection to the most probable options,
    # reducing the chance of errors. Recommended values: 0.8, 0.95
    TOP_P = float(os.getenv("TOP_P", "0.95")) 

    # UI
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7865"))
    THEME = "default"

    # Blockchain dataset (public BigQuery dataset for Ethereum)
    PUBLIC_DATABASE = "bigquery-public-data.crypto_ethereum"

    # Intent classifier (separate knobs from SQL generation)
    INTENT_MODEL = os.getenv("INTENT_MODEL", os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", "0.0"))
    INTENT_TOP_P = float(os.getenv("INTENT_TOP_P", "0.95"))
    INTENT_TOP_K = int(os.getenv("INTENT_TOP_K", "40"))
    INTENT_MAX_TOKENS = int(os.getenv("INTENT_MAX_TOKENS", "256"))
    INTENT_CANDIDATE_COUNT = int(os.getenv("INTENT_CANDIDATE_COUNT", "1"))
    INTENT_RESPONSE_MIME_TYPE = os.getenv("INTENT_RESPONSE_MIME_TYPE", "text/plain")
    INTENT_STOP_SEQUENCES = os.getenv("INTENT_STOP_SEQUENCES", "")
    INTENT_SYSTEM_INSTRUCTION = os.getenv(
        "INTENT_SYSTEM_INSTRUCTION",
        (
            "You are an intent classifier. Given a user question, "
            "respond with exactly one word: 'analytics' if the question is about data analytics, "
            "metrics, queries, time series, transactions, aggregations, or BI; otherwise respond "
            "with 'non_analytics'."
        ),
    )
    INTENT_API_VERSION = os.getenv("INTENT_API_VERSION", os.getenv("GOOGLE_GENAI_API_VERSION", "v1"))
    # If vertexai is set to "False" (case-insensitive), disable intent classifier usage
    USE_INTENT_CLASSIFIER = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() != "false"