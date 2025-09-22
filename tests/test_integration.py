# test_integration.py

import unittest
import os
from google.cloud import bigquery
from bigq_ai_app.core.prompt_builder import get_schema, build_prompt


class TestIntegration(unittest.TestCase):

    def setUp(self):
        """Set up integration test with real BigQuery client if available."""
        try:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION")
            if project_id and location:
                self.client = bigquery.Client(project=project_id, location=location)
                self.has_credentials = True
            else:
                self.has_credentials = False
        except Exception:
            self.has_credentials = False

    @unittest.skipUnless(os.getenv("GOOGLE_CLOUD_PROJECT"), "BigQuery credentials not available")
    def test_real_schema_fetching(self):
        """Test schema fetching with real BigQuery tables."""
        table_refs = [
            "bigquery-public-data.crypto_ethereum.transactions",
            "bigquery-public-data.crypto_ethereum.token_transfers"
        ]

        schema_info = get_schema(self.client, table_refs)

        # Check that we got schema information
        self.assertIn("Table:", schema_info)
        self.assertIn("bigquery-public-data.crypto_ethereum.transactions", schema_info)
        self.assertIn("bigquery-public-data.crypto_ethereum.token_transfers", schema_info)

        # Check for column information
        self.assertIn("hash", schema_info)
        self.assertIn("from_address", schema_info)

    @unittest.skipUnless(os.getenv("GOOGLE_CLOUD_PROJECT"), "BigQuery credentials not available")
    def test_real_prompt_building(self):
        """Test prompt building with real schema."""
        user_query = "Show me the top 10 transactions by value"
        table_refs = ["bigquery-public-data.crypto_ethereum.transactions"]

        prompt = build_prompt(self.client, user_query, table_refs)

        self.assertIn("You are a BigQuery expert", prompt)
        self.assertIn(user_query, prompt)
        self.assertIn("Table:", prompt)
        self.assertIn("transactions", prompt)


if __name__ == '__main__':
    unittest.main()
