# prompt_builder.py

from google.cloud import bigquery
from typing import List, Dict, Any

def get_table_schema(client: bigquery.Client, table_ref: str) -> Dict[str, Any]:
    """Fetches the schema of a BigQuery table."""
    try:
        table = client.get_table(table_ref)
        schema = {
            "table_name": table_ref,
            "description": table.description or "No description available",
            "columns": []
        }
        
        for field in table.schema:
            column_info = {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description or "No description available"
            }
            schema["columns"].append(column_info)
        
        return schema
    except Exception as e:
        return {"error": f"Failed to fetch schema for {table_ref}: {str(e)}"}

def infer_relationships(schemas: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Infers relationships between tables based on column names."""
    relationships = []
    column_map = {}
    
    # Build a map of table -> columns
    for schema in schemas:
        if "error" not in schema:
            table_name = schema["table_name"]
            columns = [col["name"] for col in schema["columns"]]
            column_map[table_name] = columns
    
    # Look for potential relationships
    for table1, cols1 in column_map.items():
        for table2, cols2 in column_map.items():
            if table1 != table2:
                for col1 in cols1:
                    for col2 in cols2:
                        # Check for common patterns
                        if (col1 == col2 or 
                            col1.replace("_", "") == col2.replace("_", "") or
                            col1 in col2 or col2 in col1):
                            # Avoid duplicates
                            rel_key = tuple(sorted([table1, table2]))
                            if rel_key not in [tuple(sorted([r["table1"], r["table2"]])) for r in relationships]:
                                relationships.append({
                                    "table1": table1,
                                    "table2": table2,
                                    "column1": col1,
                                    "column2": col2,
                                    "type": "potential_foreign_key"
                                })
    
    return relationships

def get_schema(client: bigquery.Client, table_refs: List[str]) -> str:
    """Returns the schema of the BigQuery tables with descriptions and inferred relationships."""
    schemas = []
    for table_ref in table_refs:
        schema = get_table_schema(client, table_ref)
        schemas.append(schema)
    
    # Infer relationships
    relationships = infer_relationships(schemas)
    
    # Build the schema string
    schema_parts = []
    
    for schema in schemas:
        if "error" in schema:
            schema_parts.append(f"Error fetching schema: {schema['error']}")
            continue
            
        schema_parts.append(f"Table: {schema['table_name']}")
        if schema["description"]:
            schema_parts.append(f"Description: {schema['description']}")
        
        schema_parts.append("Columns:")
        for col in schema["columns"]:
            desc = col["description"]
            schema_parts.append(f"  - {col['name']} ({col['type']}, {col['mode']}): {desc}")
        
        schema_parts.append("")
    
    # Add relationships
    if relationships:
        schema_parts.append("Inferred Relationships:")
        for rel in relationships:
            schema_parts.append(f"  - {rel['table1']} -> {rel['table2']} via {rel['column1']} = {rel['column2']}")
        schema_parts.append("")
    
    return "\n".join(schema_parts)

def build_prompt(client: bigquery.Client, user_query: str, table_refs: List[str], retrieved_context: str = "") -> str:
    """Constructs a basic LLM prompt with user query and schema info."""
    schema_info = get_schema(client, table_refs)
    # Build the human-readable prompt body
    body = f"""You are a BigQuery expert. Given the following table schema:

{schema_info}
"""
    if retrieved_context:
        body += f"\nRetrieved Context:\n{retrieved_context}\n"
    
    body += f"""
Write ONLY a BigQuery SQL query to answer the following question. Do not include any explanations, comments, or markdown formatting. Return only the raw SQL query:

{user_query}
"""

    # Wrap the body in triple single quotes so it can be embedded directly inside
    # a BigQuery ML.GENERATE_TEXT SQL call as (SELECT '''...''' AS prompt)
    # Escape any occurrence of triple-single-quotes in the body just in case.
    safe_body = body.replace("'''", "\'\'\'")
    prompt = "'''" + safe_body + "'''"
    return prompt