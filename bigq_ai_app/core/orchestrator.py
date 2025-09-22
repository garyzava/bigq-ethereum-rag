# orchestrator.py

from google.cloud import bigquery
from typing import List
from bigq_ai_app.core.prompt_builder import build_prompt
from bigq_ai_app.core.sql_generator import generate_sql
from bigq_ai_app.core.validator import validate_sql
from bigq_ai_app.core.executor import execute_sql
from bigq_ai_app.core.config import BaseConfig
from bigq_ai_app.utils.table_selection import choose_table_refs_by_keyword
from bigq_ai_app.utils.schema_reader import read_schema_catalog_api
from bigq_ai_app.utils.ai_inference import infer_descriptions_with_ai
from bigq_ai_app.core.semantic_orchestrator import run_semantic_query
from bigq_ai_app.core.multimodal_orchestrator import run_multimodal_query
import pandas as pd


def summarize_results(df: pd.DataFrame, max_rows: int = 2) -> str:
    """Create a compact plain-text summary for a DataFrame.

    The summary includes row/column counts, dtypes, and a small sample (CSV) of rows.
    This is safe to show in a plain Textbox/Log area.
    """
    if df is None or df.empty:
        return "No results returned."

    parts = []
    parts.append(f"Rows: {len(df)}")
    parts.append(f"Columns: {len(df.columns)}")
    parts.append("Column types:")
    for col, dtype in df.dtypes.items():
        parts.append(f"- {col}: {dtype}")

    parts.append("\nSample row (CSV):")
    try:
        sample_csv = df.head(max_rows).to_csv(index=False)
        parts.append(sample_csv)
    except Exception:
        # Fallback to a very small, safe representation
        parts.append(str(df.head(max_rows)))

    return "\n".join(parts)

def run_query(user_query: str, client: bigquery.Client, project_id: str, location: str, table_refs: List[str] = None) -> tuple[str, pd.DataFrame, str]:
    """Manages the agent workflow and returns (sql, results_df, notes).

    Results are returned as a pandas DataFrame so the UI can render them as a table.
    """
    notes = ""  # Initialize notes
    #initialize table_refs to nothing
    # ----- usage -----
    if table_refs is None:
        dataset = BaseConfig.PUBLIC_DATABASE
        catalog = read_schema_catalog_api(client, dataset)
        if not catalog:
            # If catalog is empty, try to infer descriptions
            table_refs = [f"{dataset}.{t}" for t in ["transactions", "blocks", "logs", "token_transfers"]]  # default tables
            dataset_id = f"{project_id}.bq_llm"
            inferred = infer_descriptions_with_ai(client, table_refs, dataset_id)
            # Use inferred if available, but for now, proceed with defaults
        else:
            table_refs = [f"{dataset}.{t}" for t in catalog.keys()]

    # Auto-pick the most relevant table(s) for this query to reduce accidental joins
    try:
        chosen_tables = choose_table_refs_by_keyword(client, table_refs, user_query, max_tables=2)
        if chosen_tables:
            table_refs = chosen_tables
    except Exception:
        # on any failure, keep the original table_refs
        pass
    notes = f"Tables Chosen: {table_refs}"
    #notes += f"Tables Chosen: {table_refs}"
    #notes += f"\n{table_refs}"

    prompt = build_prompt(client, user_query, table_refs)
    # append to notes rather than overwrite so previous messages are preserved
    notes += "\nPrompt created."

    dataset_id = f"{project_id}.bq_llm"
    sql_query = generate_sql(prompt, client, dataset_id)
    # record some metadata about the generated SQL to help diagnose truncation
    notes += "\nSQL generated."
    try:
        if isinstance(sql_query, str):
            notes += f"\nSQL length: {len(sql_query)}"
            notes += f"\nSQL head: {sql_query[:200]}"
            notes += f"\nSQL tail: {sql_query[-200:]}"
    except Exception:
        pass

    is_valid, validation_message = validate_sql(sql_query, client)
    notes += f"\n{validation_message}"

    if not is_valid:
        return sql_query, pd.DataFrame(), notes

    results = execute_sql(sql_query, client)
    notes += "\nQuery executed."

    # Append a compact summary to the notes
    try:
        summary = summarize_results(results)
        notes += f"\n\nSummary:\n{summary}"
    except Exception:
        notes += "\n\nSummary: (failed to summarize results)"

    return sql_query, results, notes

def run_orchestrator(user_query: str, client: bigquery.Client, project_id: str, location: str, phase: str = "architect", table_refs: List[str] = None, uploaded_files: List[str] = None) -> tuple[str, pd.DataFrame, str]:
    """Unified orchestrator for different phases."""
    if phase == "auto":
        # If files are uploaded, prioritize answering about the files.
        if uploaded_files:
            sql, df, notes = run_multimodal_query(user_query, client, project_id, location, uploaded_files, table_refs)
            return sql, df, "Auto: used uploaded files for multimodal answer.\n" + notes

        if BaseConfig.USE_INTENT_CLASSIFIER:
            from bigq_ai_app.core.intent_classifier import classify_intent
            intent = classify_intent(user_query)
            intent_notes = f"Intent Classified: {intent}"
        else:
            intent_notes = "Intent Classification Disabled"
            intent = 'non_analytics'  # default
        #phase = 'semantic' if intent == 'analytics' else 'semantic'
        if intent == 'analytics':
            phase = 'multimodal' #GZ UPDATE HERE
        else:
            sql = ""
            df = pd.DataFrame()
            notes = "\nTry another question, please."

        # Call the decided phase
        if phase == "architect":
            sql, df, notes = run_query(user_query, client, project_id, location, table_refs)
        elif phase == "semantic":
            sql, df, notes = run_semantic_query(user_query, client, project_id, location, table_refs)
        elif phase == "multimodal":
            sql, df, notes = run_multimodal_query(user_query, client, project_id, location, uploaded_files, table_refs)
        
        # Prepend intent notes
        notes = intent_notes + "\n" + notes
        return sql, df, notes

    if phase == "architect":
        return run_query(user_query, client, project_id, location, table_refs)
    elif phase == "semantic":
        return run_semantic_query(user_query, client, project_id, location, table_refs)
    elif phase == "multimodal":
        return run_multimodal_query(user_query, client, project_id, location, uploaded_files, table_refs)
    else:
        raise ValueError("Invalid phase")
