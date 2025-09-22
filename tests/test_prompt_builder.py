# test_prompt_builder.py

import unittest
from unittest.mock import Mock, patch
from google.cloud import bigquery
from bigq_ai_app.core.prompt_builder import get_table_schema, infer_relationships, get_schema, build_prompt


class TestPromptBuilder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=bigquery.Client)
        self.table_ref = "bigquery-public-data.crypto_ethereum.transactions"

    @patch('bigq_ai_app.core.prompt_builder.bigquery')
    def test_get_table_schema_success(self, mock_bigquery):
        """Test successful schema fetching."""
        # Mock table object
        mock_table = Mock()
        mock_table.description = "Test table description"

        # Mock schema fields
        mock_field1 = Mock()
        mock_field1.name = "hash"
        mock_field1.field_type = "STRING"
        mock_field1.mode = "REQUIRED"
        mock_field1.description = "Transaction hash"

        mock_field2 = Mock()
        mock_field2.name = "value"
        mock_field2.field_type = "NUMERIC"
        mock_field2.mode = "NULLABLE"
        mock_field2.description = "Transaction value"

        mock_table.schema = [mock_field1, mock_field2]

        self.mock_client.get_table.return_value = mock_table

        result = get_table_schema(self.mock_client, self.table_ref)

        self.assertEqual(result["table_name"], self.table_ref)
        self.assertEqual(result["description"], "Test table description")
        self.assertEqual(len(result["columns"]), 2)
        self.assertEqual(result["columns"][0]["name"], "hash")
        self.assertEqual(result["columns"][0]["type"], "STRING")
        self.assertEqual(result["columns"][0]["mode"], "REQUIRED")
        self.assertEqual(result["columns"][0]["description"], "Transaction hash")

    @patch('bigq_ai_app.core.prompt_builder.bigquery')
    def test_get_table_schema_error(self, mock_bigquery):
        """Test schema fetching with error."""
        self.mock_client.get_table.side_effect = Exception("Connection failed")

        result = get_table_schema(self.mock_client, self.table_ref)

        self.assertIn("error", result)
        self.assertIn("Connection failed", result["error"])

    def test_infer_relationships(self):
        """Test relationship inference between tables."""
        schemas = [
            {
                "table_name": "table1",
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "table_name": "table2",
                "columns": [
                    {"name": "table1_id", "type": "INTEGER"},
                    {"name": "value", "type": "FLOAT"}
                ]
            }
        ]

        relationships = infer_relationships(schemas)

        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["table1"], "table1")
        self.assertEqual(relationships[0]["table2"], "table2")
        self.assertEqual(relationships[0]["column1"], "id")
        self.assertEqual(relationships[0]["column2"], "table1_id")

    @patch('bigq_ai_app.core.prompt_builder.get_table_schema')
    def test_get_schema(self, mock_get_table_schema):
        """Test schema string generation."""
        mock_get_table_schema.side_effect = [
            {
                "table_name": "table1",
                "description": "First table",
                "columns": [
                    {"name": "id", "type": "INTEGER", "mode": "REQUIRED", "description": "Primary key"}
                ]
            },
            {
                "table_name": "table2",
                "description": "Second table",
                "columns": [
                    {"name": "table1_id", "type": "INTEGER", "mode": "NULLABLE", "description": "Foreign key"}
                ]
            }
        ]

        table_refs = ["table1", "table2"]
        result = get_schema(self.mock_client, table_refs)

        self.assertIn("Table: table1", result)
        self.assertIn("Description: First table", result)
        self.assertIn("- id (INTEGER, REQUIRED): Primary key", result)
        self.assertIn("Inferred Relationships:", result)
        self.assertIn("table1 -> table2 via id = table1_id", result)

    @patch('bigq_ai_app.core.prompt_builder.get_schema')
    def test_build_prompt(self, mock_get_schema):
        """Test prompt building."""
        mock_get_schema.return_value = "Mock schema information"

        user_query = "Show me all transactions"
        table_refs = ["table1"]

        result = build_prompt(self.mock_client, user_query, table_refs)

        self.assertIn("You are a BigQuery expert", result)
        self.assertIn("Mock schema information", result)
        self.assertIn("Show me all transactions", result)
        self.assertIn("You are a BigQuery expert", result)


if __name__ == '__main__':
    unittest.main()
