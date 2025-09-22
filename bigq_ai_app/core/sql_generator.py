# sql_generator.py

from google.cloud import bigquery
from bigq_ai_app.core.config import BaseConfig
import re
from typing import Optional


def _extract_sql_from_prompt(prompt: str) -> Optional[str]:
  """Extract SQL code from a triple-backtick block.

  Supports blocks like ```sql\nSELECT ...\n``` or generic ```...```.
  Returns the raw SQL text or None if no code block is found.
  """
  # Match ```sql ... ``` or ``` ... ``` capturing inner content (DOTALL)
  m = re.search(r"```(?:sql)?\s*(.*?)```", prompt, flags=re.DOTALL | re.IGNORECASE)
  if not m:
    return None
  return m.group(1).strip()


def _qualify_table_references(sql: str, public_db: str) -> str:
  """Ensure table identifiers after FROM/JOIN/INTO are qualified with the public_db.

  This is a conservative approach:
  - Only qualifies simple identifiers with no dot (e.g. 'transactions' -> `project.dataset.transactions`).
  - Leaves already-qualified identifiers (contain a dot) untouched.
  - Handles optional aliases after the table name.
  """

  def _qualify(match: re.Match) -> str:
    prefix = match.group('prefix')  # FROM / JOIN / INTO / UPDATE
    quote1 = match.group('q1') or ''
    name = match.group('name')
    quote2 = match.group('q2') or ''
    alias = match.group('alias') or ''

    # If the captured name already contains a dot (qualified) leave it alone.
    if '.' in name:
      return match.group(0)

    # Otherwise, qualify with the public_db and wrap in backticks
    qualified = f"`{public_db}.{name}`"
    return f"{prefix}{qualified}{alias}"

  # Patterns: FROM <name> [AS alias], JOIN <name> [alias], INTO <name>
  # Allow backticked identifiers to contain dots, hyphens, etc. For unquoted
  # identifiers match up to whitespace or a closing backtick. This avoids
  # splitting project ids like `bigquery-public-data.crypto_ethereum`.
  pattern = re.compile(
    r"(?P<prefix>\b(?:FROM|JOIN|INTO)\s+)(?P<q1>`?)(?P<name>[^`\s]+)(?P<q2>`?)(?P<alias>\s+(?:AS\s+)?[A-Za-z0-9_]+)?",
    flags=re.IGNORECASE,
  )

  return pattern.sub(_qualify, sql)


def _contains_public_db(sql: str, public_db: str) -> bool:
  """Return True if the public_db is already referenced in the SQL.

  Comparison is case-insensitive and ignores surrounding backticks.
  """
  clean = sql.replace('`', '').lower()
  return public_db.lower() in clean


def _collapse_duplicated_public_db(sql: str, public_db: str) -> str:
  """Collapse duplicated or split occurrences of public_db into a single well-formed occurrence.

  This handles cases where an LLM output accidentally repeats or splits the
  project.dataset string, e.g.:
    `bigquery-public-data.crypto_ethereum.bigquery`-public-data.crypto_ethereum.transactions`
  -> `bigquery-public-data.crypto_ethereum.transactions`
  """
  if not sql or public_db.lower() not in sql.replace('`', '').lower():
    return sql

  # Work on a cleaned, lowercase copy to find duplicates, but keep original for reconstruction
  clean = sql.replace('`', '')
  lclean = clean.lower()
  pdb = public_db.lower()

  first = lclean.find(pdb)
  second = lclean.find(pdb, first + 1)
  if first == -1 or second == -1:
    return sql

  # Build the collapsed string: keep everything up to first occurrence + public_db + remainder after second occurrence
  pre = clean[:first]
  post = clean[second + len(pdb):]
  collapsed = pre + public_db + post

  # Wrap the full public_db.table sequence in backticks to produce a valid
  # qualified identifier (e.g. `project.dataset.table`). Use a regex to find
  # public_db.<tablename> and wrap it.
  try:
    pat = re.compile(re.escape(public_db) + r"\.([A-Za-z0-9_]+)", flags=re.IGNORECASE)
    collapsed = pat.sub(lambda m: f"`{public_db}.{m.group(1)}`", collapsed, count=1)
  except re.error:
    # If regex fails for any reason, fall back to returning the collapsed string
    pass

  return collapsed


def _clean_sql_output(sql: str, public_db: str) -> str:
  """Normalize LLM/extracted SQL output:
  - Extract from triple-backtick blocks if present
  - Remove leading 'sql' token
  - Collapse duplicated/split public_db occurrences
  - Trim whitespace
  """
  if sql is None:
    return sql

  s = str(sql)

  # 1) Extract body from triple-backtick fence if present
  m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
  if m:
    s = m.group(1).strip()

  # 2) Remove leading 'sql' or 'SQL:' tokens
  s = re.sub(r"^\s*(?:sql\b[:\s]*)", "", s, flags=re.IGNORECASE)

  # 3) Collapse duplicated public_db occurrences (if any)
  s = _collapse_duplicated_public_db(s, public_db)

  return s.strip()


def generate_sql(prompt: str, client: bigquery.Client, dataset_id: str) -> str:
  """Generates or extracts a BigQuery SQL query and ensures it targets PUBLIC_DATABASE.

  Behavior:
  - If the incoming `prompt` contains a SQL code block (triple backticks), extract and return
    that SQL after qualifying table identifiers with `BaseConfig.PUBLIC_DATABASE`.
  - Otherwise, call the remote ML.GENERATE_TEXT model (keeps using `dataset_id` for model location),
    then post-process the returned SQL to qualify table identifiers with `BaseConfig.PUBLIC_DATABASE`.
  """
  temperature = BaseConfig.TEMPERATURE
  max_tokens = BaseConfig.MAX_TOKENS
  public_db = BaseConfig.PUBLIC_DATABASE

  # 1) If prompt already contains an SQL code block, extract and use it.
  extracted = _extract_sql_from_prompt(prompt)
  if extracted:
    cleaned = _clean_sql_output(extracted, public_db)
    # If the extracted SQL already references the public_db, return cleaned-as-is.
    if _contains_public_db(cleaned, public_db):
      return cleaned

    qualified = _qualify_table_references(cleaned, public_db)
    return qualified

  # 2) Otherwise, call the LLM to generate SQL and then qualify it.
  query = f"""
  SELECT
    COALESCE(
         JSON_VALUE(ml_generate_text_result, '$.candidates[0].content.parts[0].text'),
         JSON_VALUE(ml_generate_text_result, '$.predictions[0].content')
       ) AS generated_sql
  FROM ML.GENERATE_TEXT(
    MODEL `{dataset_id}.transaction_summarizer`,
    (SELECT @prompt AS prompt),
    STRUCT({temperature} AS temperature, {max_tokens} AS max_output_tokens)
  );
  """
  try:
    # Use a parameterized query to safely pass the user prompt (avoids syntax errors
    # when the prompt contains quotes, parentheses, or backticks).
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("prompt", "STRING", prompt)]
    )
    job = client.query(query, job_config=job_config)
    result = job.to_dataframe()
    if result.empty:
      return "Error: No SQL generated."

    generated = result.iloc[0]['generated_sql']
    cleaned = _clean_sql_output(generated, public_db)
    # If the generated SQL already references the public database, don't re-qualify
    if _contains_public_db(cleaned, public_db):
      return cleaned

    # Post-process to ensure the public database is used in table references
    qualified = _qualify_table_references(cleaned, public_db)
    return qualified
  except Exception as e:
    return f"Error generating SQL: {e}"
