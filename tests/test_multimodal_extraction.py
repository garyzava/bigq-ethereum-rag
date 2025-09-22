import os
from bigq_ai_app.core.multimodal_orchestrator import extract_text_from_files


def test_extract_text_from_txt(tmp_path):
    # Create a small text file
    p = tmp_path / "test_document.txt"
    p.write_text("Hello Ethereum! Latest blocks and transactions.")

    texts, notes, uris = extract_text_from_files([str(p)])
    assert isinstance(texts, list) and len(texts) == 1
    assert "Hello Ethereum" in texts[0]
    assert "Read text file" in notes
    assert uris == []
