# executor.py

from google.cloud import bigquery
import pandas as pd

def execute_sql(query: str, client: bigquery.Client) -> pd.DataFrame:
    """Executes the SQL and return results as a pandas DataFrame."""
    try:
        query_job = client.query(query)
        results = query_job.to_dataframe()
        return results
    except Exception as e:
        print(f"Query execution failed: {e}")
        return pd.DataFrame()
