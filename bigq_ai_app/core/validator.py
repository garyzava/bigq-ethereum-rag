# validator.py

from google.cloud import bigquery
import re

def validate_sql(query: str, client: bigquery.Client) -> tuple[bool, str]:
    """Performs a dry run of the SQL to estimate costs and check for errors."""
    # Preprocess query: remove code fences, leading 'sql' token, and BigQuery optimizer hints
    # 1) Extract from triple-backtick blocks if present
    m = re.search(r"```(?:sql)?\s*(.*?)```", query, flags=re.DOTALL | re.IGNORECASE)
    if m:
        query = m.group(1).strip()

    # 2) Remove leading 'sql' or 'SQL:' tokens sometimes prepended
    query = re.sub(r"^\s*(?:sql\b[:\s]*)", "", query, flags=re.IGNORECASE)

    # 3) Remove BigQuery optimizer hints of the form /*+ ... */
    #query = re.sub(r"/\*\+.*?\*/", "", query, flags=re.DOTALL)

    # Trim whitespace
    query = query.strip()

    # Basic sanity checks to catch obviously truncated or unbalanced SQL
    def _basic_sanity_check(s: str) -> tuple[bool, str]:
        # reject very short outputs
        if len(s) < 10:
            return False, "Query too short or empty."

        # common truncation hints
        trunc_markers = ["...", "[TRUNCATED]", "<truncated>"]
        for m in trunc_markers:
            if s.strip().endswith(m):
                return False, f"Query appears truncated (ends with '{m}')."

        # check balanced parentheses and quotes while ignoring escaped quotes
        paren = 0
        in_squote = False
        in_dquote = False
        esc = False
        for ch in s:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == "'" and not in_dquote:
                in_squote = not in_squote
                continue
            if ch == '"' and not in_squote:
                in_dquote = not in_dquote
                continue
            if in_squote or in_dquote:
                continue
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren -= 1
                if paren < 0:
                    return False, "More closing parentheses than opening."

        if in_squote or in_dquote:
            return False, "Unmatched quote in query."
        if paren != 0:
            return False, f"Unmatched parentheses in query (balance={paren})."

        return True, "Basic syntax checks passed."

    ok, msg = _basic_sanity_check(query)
    if not ok:
        return False, f"Invalid: {msg}"

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    max_bytes = 5000 * 1024 * 1024 * 1024  # 5 GB limit

    try:
        job = client.query(query, job_config=job_config)
        bytes_processed = job.total_bytes_processed
        if bytes_processed > max_bytes:
            return False, f"Query exceeds byte limit: {bytes_processed} > {max_bytes}"
        return True, f"Valid. Bytes: {bytes_processed}"
    except Exception as e:
        return False, f"Invalid: {e}"
