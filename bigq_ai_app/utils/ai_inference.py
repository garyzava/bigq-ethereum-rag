from google.cloud import bigquery
from typing import List, Dict, Any
import json


def infer_descriptions_with_ai(client: bigquery.Client, table_refs: List[str], dataset_id: str) -> Dict[str, Dict]:
    """Use AI to infer table and column descriptions from first 5 rows."""
    catalog = {}
    for table_ref in table_refs:
        try:
            # Get first 5 rows
            query = f"SELECT * FROM `{table_ref}` LIMIT 5"
            df = client.query(query).to_dataframe()
            if df.empty:
                continue

            sample_data = df.to_csv(index=False)
            table_name = table_ref.split('.')[-1]

            # Prompt for AI
            prompt = f"""
Based on the table name '{table_name}' and the following sample data (first 5 rows as CSV):

{sample_data}

Infer a brief description for the table and each column. Format as JSON:
{{
  "table_description": "Brief description",
  "columns": {{
    "column_name": "Brief description",
    ...
  }}
}}
"""

            # Use ML.GENERATE_TEXT
            ai_query = f"""
            SELECT JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.candidates[0].content.parts[0].text') AS response
            FROM ML.GENERATE_TEXT(
              MODEL `{dataset_id}.transaction_summarizer`,
              (SELECT @prompt AS prompt),
              STRUCT(0.2 AS temperature, 1024 AS max_output_tokens)
            )
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("prompt", "STRING", prompt)]
            )
            result = client.query(ai_query, job_config=job_config).to_dataframe()
            if not result.empty:
                response = result.iloc[0]['response']
                # Parse JSON
                try:
                    inferred = json.loads(response)
                    catalog[table_name] = {
                        "description": inferred.get("table_description", ""),
                        "columns": inferred.get("columns", {})
                    }
                except:
                    pass
        except Exception:
            pass
    return catalog
