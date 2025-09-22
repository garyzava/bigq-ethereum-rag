from google.cloud import bigquery
from typing import List, Dict, Any


def read_schema_catalog_api(client: bigquery.Client, full_dataset: str) -> Dict[str, Dict]:
    project, dataset = full_dataset.split(".", 1)
    catalog: Dict[str, Dict] = {}

    for t in client.list_tables(f"{project}.{dataset}"):
        table = client.get_table(t)  # one call per table
        cols = []

        # Handle nested RECORD fields by flattening "parent.child" names
        def add_fields(fields, prefix=""):
            for f in fields:
                cols.append({
                    "name": f"{prefix}{f.name}",
                    "type": f.field_type,
                    "mode": f.mode,
                    "description": f.description,  # <- column description
                })
                if f.field_type == "RECORD" and f.fields:
                    add_fields(f.fields, prefix=f"{prefix}{f.name}.")
        add_fields(table.schema)

        catalog[table.table_id] = {
            "table_type": table.table_type,       # TABLE / VIEW / EXTERNAL
            "description": table.description,     # <- table description
            "columns": cols,
        }
    return catalog
