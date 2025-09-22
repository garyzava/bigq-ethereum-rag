# semantic_orchestrator.py

from google.cloud import bigquery
from typing import List
from bigq_ai_app.core.prompt_builder import build_prompt
from bigq_ai_app.core.sql_generator import generate_sql
from bigq_ai_app.core.validator import validate_sql
from bigq_ai_app.core.executor import execute_sql
from bigq_ai_app.core.config import BaseConfig
import pandas as pd

def retrieve_similar_chunks(client: bigquery.Client, user_query: str, project_id: str, location: str, top_k: int = 5) -> str:
    """Embed the user query and retrieve top-k similar chunks from semantic_docs table using VECTOR_SEARCH."""
    dataset_id = f"{project_id}.bq_llm"
    table_id = f"{dataset_id}.semantic_docs"
    model_name = "text_embedding_model"
    model_path = f"{dataset_id}.{model_name}"
    
    # Use VECTOR_SEARCH for reliable retrieval, matching the notebook's approach
    search_query = f"""
    WITH query_vec AS (
      SELECT ml_generate_embedding_result AS query_embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{model_path}`,
        (SELECT @content AS content)
      )
    )
    SELECT
      search_result.base.source_file AS source_file,
      search_result.base.kind AS kind,
      search_result.base.chunk_index AS chunk_index,
      SUBSTR(search_result.base.content, 1, 200) AS content_preview,
      search_result.distance AS distance
    FROM VECTOR_SEARCH(
      TABLE `{table_id}`,
      'embedding',
      TABLE query_vec,
      'query_embedding',
      top_k => @top_k,
      distance_type => 'COSINE',
      options => '{{"use_brute_force": true}}'
    ) AS search_result
    ORDER BY distance
    """
    
    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("content", "STRING", user_query),
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            ]
        )
        results_df = client.query(search_query, job_config=job_config).to_dataframe()
        if results_df.empty:
            print("Retrieval returned 0 rows from semantic table.")
            return ""
        
        # Concatenate top chunks with sources and scores
        chunks = []
        for _, row in results_df.iterrows():
            chunks.append(f"Source: {row['source_file']} (Distance: {row['distance']:.3f})\n{row['content_preview']}")
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return ""

def run_semantic_query(user_query: str, client: bigquery.Client, project_id: str, location: str, table_refs: List[str] = None) -> tuple[str, pd.DataFrame, str]:
    """Semantic version of run_query: includes retrieval before prompt building."""
    if table_refs is None:
        dataset = BaseConfig.PUBLIC_DATABASE
        catalog = {}  # Simplified; reuse from orchestrator if needed
        table_refs = [f"{dataset}.{t}" for t in ["transactions", "blocks", "logs", "token_transfers"]]  # Common tables

    notes = f"Tables Chosen: {table_refs}"

    # Retrieval step
    retrieved_chunks = retrieve_similar_chunks(client, user_query, project_id, location)
    notes += f"\nRetrieved {len(retrieved_chunks.split('\n\n')) if retrieved_chunks else 0} chunks."

    # Build prompt with retrieved context
    prompt = build_prompt(client, user_query, table_refs, retrieved_context=retrieved_chunks)
    notes += "\nPrompt created with retrieval."

    dataset_id = f"{project_id}.bq_llm"
    sql_query = generate_sql(prompt, client, dataset_id)
    notes += "\nSQL generated."

    is_valid, validation_message = validate_sql(sql_query, client)
    notes += f"\n{validation_message}"

    if not is_valid:
        return sql_query, pd.DataFrame(), notes

    results = execute_sql(sql_query, client)
    notes += "\nQuery executed."

    try:
        summary = ""  # Reuse summarize_results from orchestrator if needed
        notes += f"\n\nSummary:\n{summary}"
    except Exception:
        notes += "\n\nSummary: (failed to summarize results)"

    return sql_query, results, notes