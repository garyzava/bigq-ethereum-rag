from google.cloud import bigquery
from typing import List, Dict, Any
import re
from collections import Counter


def choose_table_refs_by_keyword(client: bigquery.Client, candidate_tables: List[str], user_query: str, max_tables: int = 1) -> List[str]:
    """Pick the most relevant tables from candidate_tables for the user_query.

    Heuristic scoring:
    - +2 if the table name contains any keyword from the query
    - +1 if any column name contains any keyword from the query

    Returns up to `max_tables` table refs sorted by score (highest first).
    If metadata fetch fails, falls back to table-name matching only.
    """
    if not candidate_tables:
        return []

    # simple tokenizer: words and numbers
    tokens = [t.lower() for t in re.findall(r"\w+", user_query)]
    # basic stopwords to ignore
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "for",
        "to",
        "on",
        "by",
        "with",
        "from",
        "and",
        "or",
        "is",
        "are",
        "that",
    }
    tokens = [t for t in tokens if t and t not in stopwords and len(t) > 1]
    if not tokens:
        return candidate_tables[:max_tables]

    token_counts = Counter(tokens)
    scored: List[tuple[int, str]] = []

    def token_variants(t: str) -> set:
        # include basic plural/singular variants
        variants = {t}
        if t.endswith("s"):
            variants.add(t[:-1])
        else:
            variants.add(t + "s")
        return variants

    for tbl in candidate_tables:
        score = 0
        tbl_name = tbl.split(".")[-1]
        # split table name into tokens (handle underscores, camelCase heuristically)
        tbl_tokens = [x.lower() for x in re.findall(r"[A-Za-z0-9]+", tbl_name)]

        # name matches are strong signals
        for token, cnt in token_counts.items():
            variants = token_variants(token)
            if any(v in tbl_tokens for v in variants):
                score += 3 * cnt

        # try to fetch schema and match column names (weaker signal)
        try:
            table = client.get_table(tbl)
            cols = [c.name.lower() for c in table.schema]
            for token, cnt in token_counts.items():
                variants = token_variants(token)
                # count how many column names contain any variant
                col_matches = sum(1 for col in cols if any(v in col for v in variants))
                if col_matches:
                    score += 2 * col_matches * cnt
        except Exception:
            # metadata unavailable, keep only name-based score
            pass

        scored.append((score, tbl))

    scored.sort(key=lambda x: x[0], reverse=True)

    # pick top max_tables but only if they have positive score; otherwise fallback
    selected = [t for s, t in scored if s > 0]
    if not selected:
        return candidate_tables[:max_tables]

    return selected[:max_tables]
