from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json, re, uuid

# ---------- Config ----------

@dataclass
class Chunking:
    max_chars: int = 1100
    overlap: int   = 180

PROSE_CHUNK = Chunking(1100, 180)
SQL_CHUNK   = Chunking(600, 80)
ALLOWED_EXT = {".md", ".txt", ".sql"}

# ---------- Tiny helpers ----------

UUID_RE  = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
HEX_RE   = re.compile(r"\b0x[a-fA-F0-9]{6,}\b")
URL_RE   = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_IDENT = r'(?:`[^`]+`|"[^"]+"|$begin:math:display$[^$end:math:display$]+\]|[A-Za-z_][A-Za-z0-9_$]*)'
_QUAL  = rf'(?:{_IDENT}\.){{0,2}}{_IDENT}'
_AFTER = rf'(?i)\b(?:from|join|update|into|table)\s+({_QUAL})'
_DELF  = rf'(?i)\bdelete\s+from\s+({_QUAL})'

def _strip_quotes(s: str) -> str:
    s = s.strip()
    return s[1:-1] if (s[:1] in '`"[' and s[-1:] in '`"]') else s

def _extract_tables(sql: str) -> List[str]:
    out: List[str] = []
    for pat in (_AFTER, _DELF):
        for m in re.finditer(pat, sql):
            q = m.group(1).split()[0].rstrip(");").split(",")[0]
            parts = [_strip_quotes(p) for p in q.split(".")]
            if parts and not parts[0].startswith("("):
                out.append(".".join(parts))
    seen, uniq = set(), []
    for t in out:
        if t not in seen: seen.add(t); uniq.append(t)
    return uniq

def _entities(text: str) -> Dict[str, List[str]]:
    return {
        "uuids": list(set(UUID_RE.findall(text))),
        "hex_ids": list(set(HEX_RE.findall(text))),
        "urls": list(set(URL_RE.findall(text))),
        "emails": list(set(EMAIL_RE.findall(text))),
    }

def _normalize_quotes(s: str) -> str:
    return s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

def _looks_sql(body: str) -> bool:
    return bool(re.search(r"(?is)\b(select|insert|update|delete|merge|with)\b", body))

def _split_with_overlap(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text: return []
    if len(text) <= max_chars: return [text]
    step = max(1, max_chars - overlap)
    return [text[i:i+max_chars] for i in range(0, len(text), step)]

def _predict_chunks(L: int, max_chars: int, overlap: int) -> int:
    if L <= 0: return 0
    if L <= max_chars: return 1
    step = max(1, max_chars - overlap)
    return 1 + (max(0, L - max_chars + step - 1) // step)

def _dataset_name_only(dataset_id: str) -> str:
    return dataset_id.strip("`").split(".")[-1]

def _table_fqn(project_id: str, dataset_id: str, table_name: str) -> str:
    ds = _dataset_name_only(dataset_id)
    return f"{project_id}.{ds}.{table_name}"

# ---------- Parsing ----------

def parse_markdown_file(path: Path) -> Tuple[List[dict], List[dict]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    secs, cur_title, cur_buf, in_code, code_buf = [], None, [], False, []

    def flush_prose():
        nonlocal cur_buf
        t = _normalize_quotes("\n".join(cur_buf)).strip()
        if t: secs.append({"type":"prose","title":cur_title,"text":t})
        cur_buf = []

    def flush_code():
        nonlocal code_buf
        c = _normalize_quotes("\n".join(code_buf)).strip()
        if c: secs.append({"type":"code","title":cur_title,"code":c})
        code_buf = []

    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.*)\s*$", ln)
        if not in_code and m:
            flush_prose(); cur_title = m.group(2).strip(); continue
        if ln.startswith("```"):
            if not in_code: in_code = True; code_buf = []
            else: in_code = False; flush_code()
            continue
        (code_buf if in_code else cur_buf).append(ln)
    if in_code: flush_code()
    flush_prose()

    sql_recs, prose_recs = [], []
    for s in secs:
        if s["type"] == "prose":
            prose_recs.append({"kind":"prose","title":s["title"],"text":s["text"]})
        else:
            body = s["code"]
            if _looks_sql(body):
                ctx = ""
                for p in reversed(prose_recs):
                    if p["title"] == s["title"]:
                        ctx = p["text"]; break
                sql_recs.append({
                    "kind":"sql",
                    "title":s["title"],
                    "context": ctx.strip(),
                    "sql": body,
                    "tables": _extract_tables(body),
                    "entities": _entities(body + "\n" + ctx),
                })
    for p in prose_recs: p["entities"] = _entities(p["text"])
    for r in sql_recs + prose_recs: r["source_file"] = str(path)
    return sql_recs, prose_recs

def parse_docs_folder(docs_dir: Path) -> Tuple[List[dict], List[dict], List[Path]]:
    all_paths = [p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXT and not p.name.startswith(".")]
    sql_records, prose_records = [], []
    for path in all_paths:
        if path.suffix.lower() == ".sql":
            code = _normalize_quotes(path.read_text(encoding="utf-8", errors="ignore"))
            sql_records.append({
                "kind":"sql","title":path.stem,"context":"",
                "sql":code,"tables":_extract_tables(code),"entities":_entities(code),"source_file":str(path)
            })
        else:
            s, p = parse_markdown_file(path); sql_records += s; prose_records += p
    return sql_records, prose_records, all_paths

# ---------- Dry run ----------

def dry_run_summary(sql_records: Sequence[dict], prose_records: Sequence[dict],
                    prose_chunk: Chunking = PROSE_CHUNK, sql_chunk: Chunking = SQL_CHUNK,
                    filter_filenames: Optional[Sequence[str]] = None) -> None:
    want = set(filter_filenames) if filter_filenames else None
    def keep(path: str) -> bool: return True if not want else (Path(path).name in want)

    buckets: Dict[str, Dict[str,int]] = {}
    def bump(src: str, kind: str, chunks: int, chars: int):
        d = buckets.setdefault(src, {"prose_docs":0,"prose_chunks":0,"prose_chars":0,"sql_docs":0,"sql_chunks":0,"sql_chars":0})
        d[f"{kind}_docs"] += 1; d[f"{kind}_chunks"] += chunks; d[f"{kind}_chars"] += chars

    for r in prose_records:
        if keep(r["source_file"]):
            body = f"# {r.get('title','')}\n\n{r.get('text','')}"
            bump(r["source_file"], "prose", _predict_chunks(len(body), prose_chunk.max_chars, prose_chunk.overlap), len(body))
    for r in sql_records:
        if keep(r["source_file"]):
            ctx, sql = r.get("context",""), r.get("sql","")
            body = f"# {r.get('title','')}\n\n" + (f"**Context**\n\n{ctx}\n\n" if ctx else "") + f"**SQL**\n\n```sql\n{sql}\n```"
            bump(r["source_file"], "sql", _predict_chunks(len(body), sql_chunk.max_chars, sql_chunk.overlap), len(body))

    print("\n=== DRY RUN: Predicted chunks per file (no writes) ===")
    if not buckets: print("No matching files."); print("=== END DRY RUN ==="); return
    total_docs = total_chunks = 0
    for src in sorted(buckets):
        b = buckets[src]; docs = b["prose_docs"]+b["sql_docs"]; chunks = b["prose_chunks"]+b["sql_chunks"]
        print(f"\nFile: {src}")
        print(f"  Prose: docs={b['prose_docs']}, chunks={b['prose_chunks']}")
        print(f"  SQL  : docs={b['sql_docs']}, chunks={b['sql_chunks']}")
        print(f"  TOTAL: docs={docs}, chunks={chunks}, avg_chunks_per_doc={(chunks/docs if docs else 0):.2f}")
        total_docs += docs; total_chunks += chunks
    if total_docs:
        print("\n=== Overall ===")
        print(f"  docs={total_docs}, chunks={total_chunks}, avg_chunks_per_doc={(total_chunks/total_docs):.2f}")
    print("=== END DRY RUN ===\n")

# ---------- BigQuery ops (location passed to client.query) ----------

def ensure_dataset(client, project_id: str, location: str, dataset_id: str) -> str:
    from google.cloud import bigquery
    ds_name = _dataset_name_only(dataset_id)
    ds = bigquery.Dataset(f"{project_id}.{ds_name}")
    ds.location = location
    try:
        client.get_dataset(ds)
    except Exception:
        client.create_dataset(ds)
    return ds.dataset_id

def reset_table(client, project_id: str, location: str, dataset_id: str, table_name: str) -> str:
    """DROP + CREATE with fixed schema (call once)."""
    table_fqn = _table_fqn(project_id, dataset_id, table_name)
    # Separate statements for max compatibility
    client.query(f"DROP TABLE IF EXISTS `{table_fqn}`", location=location).result()
    create_sql = f"""
    CREATE TABLE `{table_fqn}` (
      id STRING,
      doc_id STRING,
      source_file STRING,
      title STRING,
      kind STRING,
      chunk_index INT64,
      content STRING,
      metadata STRING
    )
    """
    client.query(create_sql, location=location).result()
    print(f"Table reset: `{table_fqn}`")
    return table_fqn

def truncate_table(client, project_id: str, location: str, dataset_id: str, table_name: str) -> str:
    table_fqn = _table_fqn(project_id, dataset_id, table_name)
    client.query(f"TRUNCATE TABLE `{table_fqn}`", location=location).result()
    print(f"Table truncated: `{table_fqn}`")
    return table_fqn

def insert_rows(client, table_fqn: str, rows: List[dict]) -> None:
    clean = [{k: v for k, v in r.items() if v is not None} for r in rows]
    for i in range(0, len(clean), 500):
        batch = clean[i:i+500]
        errors = client.insert_rows_json(table_fqn, batch)
        if errors:
            raise RuntimeError(f"BigQuery insert errors at batch {i//500}: {errors}")

# ---------- Orchestrator ----------

def ingest_docs_folder(
    client,
    project_id: str,
    location: str,
    docs_dir: Path,
    dataset_id: str,
    table_name: str,
    dry_run: bool = True,
    filter_filenames: Optional[Sequence[str]] = None,
    prose_chunk: Chunking = PROSE_CHUNK,
    sql_chunk: Chunking = SQL_CHUNK,
    reset_table_once: bool = False,       # first run: True (DROP+CREATE)
    truncate_before_insert: bool = True   # subsequent runs: True (TRUNCATE)
) -> None:
    sql_records, prose_records, _ = parse_docs_folder(docs_dir)
    dry_run_summary(sql_records, prose_records, prose_chunk, sql_chunk, filter_filenames)
    if dry_run:
        print("Dry run only. Set dry_run=False to write.")
        return

    ensure_dataset(client, project_id, location, dataset_id)
    if reset_table_once:
        table_fqn = reset_table(client, project_id, location, dataset_id, table_name)
    else:
        table_fqn = _table_fqn(project_id, dataset_id, table_name)
        if truncate_before_insert:
            truncate_table(client, project_id, location, dataset_id, table_name)

    rows: List[dict] = []

    # prose
    for r in prose_records:
        body = f"# {r.get('title','')}\n\n{r.get('text','')}"
        chunks = _split_with_overlap(body, prose_chunk.max_chars, prose_chunk.overlap)
        meta = json.dumps({"source_file": r["source_file"], "entities": r["entities"], "kind": "prose"}, ensure_ascii=False)
        doc_id = str(uuid.uuid4())
        for idx, c in enumerate(chunks):
            rows.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "source_file": r["source_file"],
                "title": r.get("title", ""),
                "kind": "prose",
                "chunk_index": idx,
                "content": c,
                "metadata": meta,
            })

    # sql
    for r in sql_records:
        ctx, sql = r.get("context",""), r.get("sql","")
        body = f"# {r.get('title','')}\n\n" + (f"**Context**\n\n{ctx}\n\n" if ctx else "") + f"**SQL**\n\n```sql\n{sql}\n```"
        chunks = _split_with_overlap(body, sql_chunk.max_chars, sql_chunk.overlap)
        meta = json.dumps({"source_file": r["source_file"], "entities": r["entities"], "tables": r["tables"], "kind": "sql"}, ensure_ascii=False)
        doc_id = str(uuid.uuid4())

        # GEMINI Add this print statement to see if chunking is working BEFORE insertion
        print(f"DEBUG: File '{Path(r['source_file']).name}' was split into {len(chunks)} chunks.")


        for idx, c in enumerate(chunks):
            rows.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "source_file": r["source_file"],
                "title": r.get("title",""),
                "kind": "sql",
                "chunk_index": idx,
                "content": c,
                "metadata": meta,
            })

    #insert_rows(client, table_fqn, rows)
    insert_rows_as_batch(client, table_fqn, rows) # GEMINI new test
    print(f"Ingested {len(rows)} chunks into `{table_fqn}`.")

# ---------- Embeddings (optional) ----------

def create_embedding_model(client, project_id: str, location: str,
                           dataset_id: str,
                           model_name: str = "text_embedding_model",
                           connection_name: str = "bq-llm-connection",
                           endpoint: str = "text-embedding-004") -> str:
    table_ds = _dataset_name_only(dataset_id)
    model_path = f"{project_id}.{table_ds}.{model_name}"
    connection_resource = f"projects/{project_id}/locations/{location}/connections/{connection_name}"
    ddl = f"""
    CREATE OR REPLACE MODEL `{model_path}`
    REMOTE WITH CONNECTION `{connection_resource}`
    OPTIONS ( ENDPOINT = '{endpoint}' );
    """
    client.query(ddl, location=location).result()
    print(f"Embedding model ready: `{model_path}` (endpoint={endpoint})")
    return model_path

def upsert_embeddings_for_new_rows_bkp(client, project_id: str, location: str,
                                   dataset_id: str, table_name: str,
                                   model_path: str,
                                   content_col: str = "content",
                                   embedding_col: str = "embedding") -> None:
    table_fqn = _table_fqn(project_id, dataset_id, table_name)
    client.query(
        f"ALTER TABLE `{table_fqn}` ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>",
        location=location
    ).result()
    client.query(
        f"""
        UPDATE `{table_fqn}` t
        SET {embedding_col} = ML.GENERATE_EMBEDDING(MODEL `{model_path}`, t.{content_col})
        WHERE {embedding_col} IS NULL AND t.{content_col} IS NOT NULL
        """,
        location=location
    ).result()
    print(f"Embeddings upserted for `{table_fqn}` using `{model_path}`.")

def upsert_embeddings_for_new_rows_chatgptNOTWORKING(client, project_id: str, location: str,
                                   dataset_id: str, table_name: str,
                                   model_path: str,
                                   content_col: str = "content",
                                   embedding_col: str = "embedding",
                                   id_col: str = "id") -> None:
    """
    Adds {embedding_col} if missing and fills NULLs by joining the
    table-valued result of ML.GENERATE_EMBEDDING back on {id_col}.
    """
    table_fqn = _table_fqn(project_id, dataset_id, table_name)

    # 1) Ensure embedding column exists
    client.query(
        f"ALTER TABLE `{table_fqn}` ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>",
        location=location
    ).result()

    # 2) Update using the table-valued function + join for portability
    update_sql = f"""
    UPDATE `{table_fqn}` AS t
    SET {embedding_col} = s.embedding
    FROM (
      SELECT {id_col} AS id, embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{model_path}`,
        TABLE (
          SELECT {id_col}, {content_col} AS content
          FROM `{table_fqn}`
          WHERE {embedding_col} IS NULL AND {content_col} IS NOT NULL
        )
      )
    ) AS s
    WHERE t.{id_col} = s.id
    """
    client.query(update_sql, location=location).result()
    print(f"Embeddings upserted for `{table_fqn}` using `{model_path}`.")    

def upsert_embeddings_for_new_rows_WORKS(client, project_id: str, location: str,
                                   dataset_id: str, table_name: str,
                                   model_path: str,
                                   content_col: str = "content",
                                   embedding_col: str = "embedding",
                                   id_col: str = "id") -> None:
    """
    Adds {embedding_col} if missing and fills NULLs by joining the
    table-valued result of ML.GENERATE_EMBEDDING back on {id_col}.
    """
    table_fqn = _table_fqn(project_id, dataset_id, table_name)

    # 1) Ensure embedding column exists
    client.query(
        f"ALTER TABLE `{table_fqn}` ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>",
        #f"ALTER TABLE `{table_fqn}` ADD COLUMN IF NOT EXISTS {embedding_col} FLOAT64",
        location=location
    ).result()

    # 2) Update using the table-valued function and a JOIN
    update_sql = f"""
    MERGE `{table_fqn}` AS t
    USING (
      SELECT
        {id_col} as id,
        --ml_generate_embedding_result.embeddings[0].embedding as embedding
        ml_generate_embedding_result as embedding
      FROM
        ML.GENERATE_EMBEDDING(
          MODEL `{model_path}`,
          (
            SELECT {id_col}, {content_col}
            FROM `{table_fqn}`
            WHERE {embedding_col} IS NULL AND {content_col} IS NOT NULL
          )
        )
    ) AS s
    ON t.{id_col} = s.id
    WHEN MATCHED THEN
      UPDATE SET t.{embedding_col} = s.embedding
    """
    client.query(update_sql, location=location).result()
    print(f"Embeddings upserted for `{table_fqn}` using `{model_path}`.")    

def upsert_embeddings_for_new_rows(client, project_id: str, location: str,
                                   dataset_id: str, table_name: str,
                                   model_path: str,
                                   content_col: str = "content",
                                   embedding_col: str = "embedding",
                                   id_col: str = "id",
                                   batch_size: int = 20) -> None:
    """
    Finds all rows without embeddings and processes them in small batches
    to avoid hitting API rate limits.
    """
    table_fqn = _table_fqn(project_id, dataset_id, table_name)

    # 1. First, ensure the embedding column exists
    client.query(
        f"ALTER TABLE `{table_fqn}` ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>",
        location=location
    ).result()

    # 2. Get a list of all row IDs that need to be processed
    ids_to_process_query = f"""
    SELECT {id_col} FROM `{table_fqn}`
    #WHERE {embedding_col} IS NULL AND {content_col} IS NOT NULL AND TRIM({content_col}) != ''
    WHERE (embedding IS NULL OR ARRAY_LENGTH(embedding) = 0) AND TRIM(content) != ''
    """
    query_job = client.query(ids_to_process_query, location=location)
    ids_to_process = [row[id_col] for row in query_job.result()]

    if not ids_to_process:
        print("No new rows to embed.")
        return

    print(f"Found {len(ids_to_process)} rows to embed. Processing in batches of {batch_size}...")

    # 3. Loop through the IDs in small batches
    for i in range(0, len(ids_to_process), batch_size):
        batch_ids = ids_to_process[i:i + batch_size]
        
        # Friendly progress indicator
        batch_num = (i // batch_size) + 1
        total_batches = (len(ids_to_process) + batch_size - 1) // batch_size
        print(f"  - Processing batch {batch_num} of {total_batches}...")

        # Format the list of IDs for the SQL IN clause
        formatted_ids = ", ".join([f"'{_id}'" for _id in batch_ids])

        update_sql = f"""
        MERGE `{table_fqn}` AS t
        USING (
          SELECT
            {id_col} as id,
            ml_generate_embedding_result as embedding
          FROM
            ML.GENERATE_EMBEDDING(
              MODEL `{model_path}`,
              (
                SELECT {id_col}, {content_col}
                FROM `{table_fqn}`
                WHERE {id_col} IN ({formatted_ids}) -- Process only IDs in the current batch
              ),
              STRUCT(TRUE AS flatten_json_output)
            )
        ) AS s
        ON t.{id_col} = s.id
        WHEN MATCHED THEN
          UPDATE SET t.{embedding_col} = s.embedding
        """
        client.query(update_sql, location=location).result()

    print(f"✅ Embeddings successfully upserted for {len(ids_to_process)} rows in `{table_fqn}`.")


def debug_embedding_row_by_row(client, project_id: str, location: str,
                               dataset_id: str, table_name: str,
                               model_path: str, num_rows_to_test: int = 5):
    """
    Fetches a few rows and tries to embed them one-by-one using a simple
    SELECT query, printing the result for each.
    """
    table_fqn = _table_fqn(project_id, dataset_id, table_name)
    
    # 1. Get a few rows that need embedding
    # --- THIS QUERY IS NOW FIXED ---
    ids_to_test_query = f"""
    SELECT id FROM `{table_fqn}`
    WHERE (embedding IS NULL OR ARRAY_LENGTH(embedding) = 0) AND TRIM(content) != ''
    LIMIT {num_rows_to_test}
    """
    print(f"--- Starting Row-by-Row Debug of {num_rows_to_test} Rows ---")
    try:
        test_ids = [row['id'] for row in client.query(ids_to_test_query, location=location).result()]
    except Exception as e:
        print(f"Could not fetch IDs to test. Error: {e}")
        return

    if not test_ids:
        print("No rows found to test.")
        return

    # 2. Loop and test each one individually
    for row_id in test_ids:
        print(f"\n--- Testing Row ID: {row_id} ---")
        
        # 3. Run the simple SELECT query
        embedding_query = f"""
        SELECT ml_generate_embedding_result
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{model_path}`,
            (
              SELECT content FROM `{table_fqn}` WHERE id = '{row_id}'
            )
        )
        """
        try:
            query_job = client.query(embedding_query, location=location)
            results = list(query_job.result())
            
            if results and results[0].ml_generate_embedding_result:
                embedding = results[0].ml_generate_embedding_result
                print(f"✅ SUCCESS: Got embedding with {len(embedding)} dimensions.")
                print(f"   Sample: {embedding[:4]}...")
            else:
                print("❌ FAILURE: Query ran but returned an empty or NULL result.")

        except Exception as e:
            print(f"❌ ERROR: The query failed with an exception: {e}")
    
    print("\n--- Debug Finished ---")



#####gemini latest not so sure about it:

# Add this new function to ingestion.py
# You will also need to `import pandas` and `from google.cloud import bigquery`
import pandas
from google.cloud import bigquery
from typing import List, Dict

def insert_rows_as_batch(client, table_fqn: str, rows: List[Dict]) -> None:
    """Inserts rows using a batch load job to avoid the streaming buffer."""
    if not rows:
        print("No rows to insert.")
        return

    df = pandas.DataFrame(rows)

    # Define the schema. This ensures data types are correct during the load.
    schema = [
        bigquery.SchemaField("id", "STRING"),
        bigquery.SchemaField("doc_id", "STRING"),
        bigquery.SchemaField("source_file", "STRING"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("kind", "STRING"),
        bigquery.SchemaField("chunk_index", "INT64"),
        bigquery.SchemaField("content", "STRING"),
        bigquery.SchemaField("metadata", "STRING"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        # This single setting handles truncating the table before loading new data.
        write_disposition="WRITE_TRUNCATE",
    )

    print(f"Starting batch load of {len(df)} rows into `{table_fqn}`...")
    
    job = client.load_table_from_dataframe(
        df, table_fqn, job_config=job_config
    )
    job.result()  # Wait for the load job to complete.
    
    print(f"Batch load complete. {len(df)} rows loaded into `{table_fqn}`.")