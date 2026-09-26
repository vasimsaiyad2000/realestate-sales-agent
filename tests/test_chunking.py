import pytest

from app.services.chunking import chunk_text


def test_chunk_text_keeps_order_and_context() -> None:
    chunks = chunk_text("one two three four five six", chunk_size=13, overlap=4)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].content == "one two three"
    assert "three" in chunks[1].content


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=10)
