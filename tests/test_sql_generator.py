import pytest
from bigq_ai_app.core.sql_generator import (
    _qualify_table_references,
    _contains_public_db,
    _extract_sql_from_prompt,
)
from bigq_ai_app.core.config import BaseConfig


def test_qualify_unqualified_name():
    sql = "SELECT * FROM transactions WHERE x=1"
    out = _qualify_table_references(sql, BaseConfig.PUBLIC_DATABASE)
    expected = f"SELECT * FROM `{BaseConfig.PUBLIC_DATABASE}.transactions` WHERE x=1"
    assert out == expected


def test_qualify_with_alias():
    sql = "SELECT t.hash FROM transactions t JOIN blocks b ON t.block=b.id"
    out = _qualify_table_references(sql, BaseConfig.PUBLIC_DATABASE)
    expected = (
        "SELECT t.hash FROM `{db}.transactions` t JOIN `{db}.blocks` b ON t.block=b.id"
    ).format(db=BaseConfig.PUBLIC_DATABASE)
    assert out == expected


def test_not_modify_already_qualified_backticked():
    sql = f"SELECT * FROM `{BaseConfig.PUBLIC_DATABASE}.transactions` ORDER BY block_timestamp DESC LIMIT 1"
    out = _qualify_table_references(sql, BaseConfig.PUBLIC_DATABASE)
    assert out == sql


def test_contains_public_db_true_false():
    sql = f"FROM `{BaseConfig.PUBLIC_DATABASE}.transactions`"
    assert _contains_public_db(sql, BaseConfig.PUBLIC_DATABASE)
    assert not _contains_public_db("SELECT * FROM transactions", BaseConfig.PUBLIC_DATABASE)


def test_extract_sql_from_prompt():
    p = "Some text\n```sql\nSELECT * FROM transactions\n```\nmore"
    extracted = _extract_sql_from_prompt(p)
    assert extracted.strip().upper().startswith("SELECT * FROM TRANSACTIONS")


def test_collapse_duplicated_public_db():
    bad = "SELECT * FROM `bigquery-public-data.crypto_ethereum.bigquery`-public-data.crypto_ethereum.transactions` ORDER BY block_timestamp DESC LIMIT 1"
    from bigq_ai_app.core.sql_generator import _collapse_duplicated_public_db
    out = _collapse_duplicated_public_db(bad, BaseConfig.PUBLIC_DATABASE)
    expected = f"SELECT * FROM `{BaseConfig.PUBLIC_DATABASE}.transactions` ORDER BY block_timestamp DESC LIMIT 1"
    assert out == expected


def test_update_statements_not_modified():
    sql = "UPDATE transactions SET value=0 WHERE id=1"
    from bigq_ai_app.core.sql_generator import _qualify_table_references
    out = _qualify_table_references(sql, BaseConfig.PUBLIC_DATABASE)
    # Should not qualify UPDATE statements per new rule
    assert out == sql
