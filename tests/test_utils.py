import pytest
from unittest.mock import Mock, patch
from bigq_ai_app.utils.table_selection import choose_table_refs_by_keyword
from bigq_ai_app.utils.schema_reader import read_schema_catalog_api
from bigq_ai_app.utils.ai_inference import infer_descriptions_with_ai

def test_choose_table_refs_by_keyword():
    client = Mock()
    candidate_tables = ["dataset.table1", "dataset.table2", "dataset.transactions"]
    user_query = "show transactions"

    # Mock get_table
    mock_table = Mock()
    mock_table.schema = [Mock(name="hash"), Mock(name="value")]
    client.get_table.return_value = mock_table

    result = choose_table_refs_by_keyword(client, candidate_tables, user_query, max_tables=1)
    assert "dataset.transactions" in result

def test_read_schema_catalog_api():
    client = Mock()
    mock_table = Mock()
    mock_table.table_id = "transactions"
    mock_table.table_type = "TABLE"
    mock_table.description = "Transaction data"
    mock_table.schema = []  # Mock as list
    client.list_tables.return_value = [mock_table]
    client.get_table.return_value = mock_table

    # Since schema is empty, it will add nothing, but test the structure
    result = read_schema_catalog_api(client, "project.dataset")
    assert "transactions" in result
    assert result["transactions"]["description"] == "Transaction data"

@patch('bigq_ai_app.utils.ai_inference.bigquery')
def test_infer_descriptions_with_ai(mock_bq):
    client = Mock()
    table_refs = ["project.dataset.transactions"]
    dataset_id = "project.bq_llm"

    # Mock query results
    mock_result = Mock()
    mock_result.to_dataframe.return_value = Mock()
    client.query.return_value = mock_result

    result = infer_descriptions_with_ai(client, table_refs, dataset_id)
    # Since it's mocked, just check it runs without error
    assert isinstance(result, dict)
