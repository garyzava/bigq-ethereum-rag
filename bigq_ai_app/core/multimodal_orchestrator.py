# multimodal_orchestrator.py

from google.cloud import bigquery
from typing import List, Tuple
from bigq_ai_app.core.prompt_builder import build_prompt
from bigq_ai_app.core.sql_generator import generate_sql
from bigq_ai_app.core.validator import validate_sql
from bigq_ai_app.core.executor import execute_sql
from bigq_ai_app.core.config import BaseConfig
import pandas as pd
import re
import os
import uuid

# Optional cloud imports (graceful fallback if unavailable)
try:  # type: ignore
    from google.cloud import vision
    from google.cloud import storage
except Exception:  # pragma: no cover - environment may not have these
    vision = None
    storage = None

def _gcs_upload(local_path: str, bucket_name: str, prefix: str) -> str:
    """Upload a local file to GCS and return the gs:// URI. Requires google-cloud-storage.

    If storage client isn't available or bucket missing, raises Exception.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage is not available")
    client = storage.Client(project=BaseConfig.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    if not bucket.exists():  # may require permissions
        raise RuntimeError(f"GCS bucket not found: {bucket_name}")
    base = os.path.basename(local_path)
    blob_path = f"{prefix.rstrip('/')}/{uuid.uuid4().hex}_{base}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_path}"

def _vision_ocr_pdf_gcs(gcs_pdf_uri: str, output_prefix: str) -> str:
    """Run async Vision OCR for a PDF in GCS. Returns concatenated text from JSON outputs.

    Follows https://cloud.google.com/vision/docs/pdf#vision_text_detection_pdf_gcs-python
    Requires google-cloud-vision and google-cloud-storage.
    """
    if vision is None or storage is None:
        raise RuntimeError("google-cloud-vision/storage not available")

    out_bucket_name, out_prefix_path = _parse_gs_uri(output_prefix)
    out_client = storage.Client(project=BaseConfig.PROJECT_ID)
    out_bucket = out_client.bucket(out_bucket_name)

    image_client = vision.ImageAnnotatorClient()
    gcs_source = vision.GcsSource(uri=gcs_pdf_uri)
    gcs_destination = vision.GcsDestination(uri=output_prefix)

    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )
    operation = image_client.async_batch_annotate_files(requests=[async_request])
    operation.result(timeout=300)  # wait up to 5 minutes

    # Collect JSON results from output GCS prefix
    texts: list[str] = []
    for blob in out_bucket.list_blobs(prefix=out_prefix_path):
        if not blob.name.lower().endswith(".json"):
            continue
        data = blob.download_as_bytes()
        try:
            import json
            response = json.loads(data)
            for resp in response.get("responses", []):
                full_text = resp.get("fullTextAnnotation", {}).get("text", "")
                if full_text:
                    texts.append(full_text)
        except Exception:
            continue
    return "\n\n".join(texts)

def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    assert gs_uri.startswith("gs://"), f"Invalid GCS URI: {gs_uri}"
    parts = gs_uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def extract_text_from_files(uploaded_files: List[str]) -> Tuple[List[str], str, List[str]]:
    """Extract text from uploaded files.

    - For PDFs: if GCS bucket configured and Vision available, upload and OCR; else try local pdfplumber; else fail gracefully.
    - For text-like files: read as UTF-8.

    Returns (texts_per_file, notes, gcs_uris) where gcs_uris are provided for files uploaded to GCS.
    """
    notes: list[str] = []
    texts: list[str] = []
    gcs_uris: list[str] = []

    bucket = (BaseConfig.GCS_BUCKET or "").strip()
    out_prefix = (BaseConfig.VISION_OUTPUT_PREFIX or "vision_ocr_output/").strip()

    # Build a concrete output prefix URI if bucket configured
    out_prefix_uri = f"gs://{bucket}/{out_prefix.rstrip('/')}/{uuid.uuid4().hex}/" if bucket else ""

    for path in uploaded_files:
        try:
            lower = path.lower()
            if lower.endswith(".pdf"):
                if bucket and storage is not None and vision is not None:
                    gs_uri = _gcs_upload(path, bucket, prefix="uploads")
                    gcs_uris.append(gs_uri)
                    ocr_text = _vision_ocr_pdf_gcs(gs_uri, out_prefix_uri)
                    if not ocr_text:
                        notes.append(f"Vision OCR returned empty text for {os.path.basename(path)}")
                    else:
                        notes.append(f"Vision OCR extracted text from {os.path.basename(path)}")
                    texts.append(ocr_text)
                else:
                    # Local fallback using pdfplumber if available
                    try:
                        import pdfplumber  # type: ignore
                        content = ""
                        with pdfplumber.open(path) as pdf:
                            for page in pdf.pages:
                                content += page.extract_text() or ""
                        texts.append(content)
                        notes.append(f"Extracted text locally from PDF: {os.path.basename(path)}")
                    except Exception as e2:
                        texts.append("")
                        notes.append(f"Failed to extract PDF {os.path.basename(path)}: {e2}")
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                texts.append(content)
                notes.append(f"Read text file: {os.path.basename(path)}")
        except Exception as e:
            texts.append("")
            notes.append(f"Error reading {os.path.basename(path)}: {e}")

    return texts, "\n".join(notes), gcs_uris

def retrieve_similar_chunks_hybrid(client: bigquery.Client, user_query: str, project_id: str, location: str, top_k: int = 5) -> str:
    """Embed the user query and retrieve top-k similar chunks using VECTOR_SEARCH with hybrid scoring (vector + keyword)."""
    dataset_id = f"{project_id}.bq_llm"
    table_id = f"{dataset_id}.semantic_docs"
    model_name = "text_embedding_model"
    model_path = f"{dataset_id}.{model_name}"
    
    # Use VECTOR_SEARCH for reliable retrieval, matching the semantic approach
    search_query = f"""
    WITH query_vec AS (
      SELECT ml_generate_embedding_result AS query_embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{model_path}`,
        (SELECT @content AS content)
      )
    )
        SELECT
            search_result.base.content AS content,
            search_result.base.source_file AS source_file,
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
        
        # For hybrid, add keyword bonus if needed, but for now, just use vector distance
        chunks = []
        for _, row in results_df.iterrows():
            chunks.append(f"Source: {row['source_file']} (Distance: {row['distance']:.3f})\n{row['content']}")
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return ""

def build_multimodal_prompt(client: bigquery.Client, user_query: str, table_refs: List[str], retrieved_context: str, uploaded_files: List[str] = None) -> str:
    """Build prompt with multimodal support: include ObjectRef for uploaded files."""
    base_prompt = build_prompt(client, user_query, table_refs, retrieved_context=retrieved_context)
    
    if uploaded_files:
        # Include a succinct mention of uploaded files; content already in retrieved_context
        names = [os.path.basename(p) for p in uploaded_files]
        multimodal_addon = "\n\nUploaded Files: " + ", ".join(names)
        base_prompt += multimodal_addon
    
    return base_prompt

def run_multimodal_query(user_query: str, client: bigquery.Client, project_id: str, location: str, uploaded_files: List[str] = None, table_refs: List[str] = None) -> tuple[str, pd.DataFrame, str]:
    """Multimodal version: if files uploaded, answer about files (Vision/GCS or local fallback); else delegate to semantic."""
    if table_refs is None:
        dataset = BaseConfig.PUBLIC_DATABASE
        table_refs = [f"{dataset}.{t}" for t in ["transactions", "blocks", "logs", "token_transfers"]]

    notes = f"Tables Chosen: {table_refs}"

    if uploaded_files:
        texts, extract_notes, gcs_uris = extract_text_from_files(uploaded_files)
        # Trim long texts to keep prompts small
        parts = []
        for i, (p, t) in enumerate(zip(uploaded_files, texts), start=1):
            preview = (t or "").strip()
            if len(preview) > 4000:
                preview = preview[:4000] + "..."
            parts.append(f"File {i}: {os.path.basename(p)}\n{preview}")
        retrieved_context = "\n\n".join(parts)
        notes += "\n" + extract_notes
        # Optional: summarize with BigFrames if available and URIs exist
        if gcs_uris:
            try:
                import bigframes.pandas as bpd  # type: ignore
                bf = bpd.DataFrame({"gs_uri": gcs_uris, "text_len": [len(t or "") for t in texts]})
                head = bf.head(3)
                notes += f"\nBigFrames summary: rows={len(bf)}, head=\n{head.to_string(index=False)}"
            except Exception:
                pass
    else:
        # Delegate to semantic pipeline for reliability
        try:
            from bigq_ai_app.core.semantic_orchestrator import run_semantic_query  # local import to avoid cycles
            sql, df, sem_notes = run_semantic_query(user_query, client, project_id, location, table_refs)
            return sql, df, notes + "\n(No files uploaded — used semantic retrieval)\n" + sem_notes
        except Exception:
            # Fallback: minimal hybrid retrieval
            retrieved_context = retrieve_similar_chunks_hybrid(client, user_query, project_id, location)
            notes += f"\nSemantic fallback failed; used hybrid retrieval with {len(retrieved_context.split('\n\n')) if retrieved_context else 0} chunks."

    # Build multimodal prompt
    prompt = build_multimodal_prompt(client, user_query, table_refs, retrieved_context, uploaded_files)
    notes += "\nMultimodal prompt built."

    dataset_id = f"{project_id}.bq_llm"
    sql_query = generate_sql(prompt, client, dataset_id)
    notes += "\nSQL generated."

    is_valid, validation_message = validate_sql(sql_query, client)
    notes += f"\n{validation_message}"

    if not is_valid:
        return sql_query, pd.DataFrame(), notes

    results = execute_sql(sql_query, client)
    notes += "\nQuery executed."

    return sql_query, results, notes
