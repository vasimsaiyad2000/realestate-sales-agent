from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    content: str
    chunk_index: int


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 180) -> list[TextChunk]:
    """Split brochure text on word boundaries while retaining retrieval context."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(words):
        current: list[str] = []
        length = 0
        end = start
        while end < len(words):
            word_length = len(words[end]) + (1 if current else 0)
            if current and length + word_length > chunk_size:
                break
            current.append(words[end])
            length += word_length
            end += 1

        chunks.append(TextChunk(" ".join(current), index))
        index += 1
        if end == len(words):
            break

        overlap_length = 0
        next_start = end
        while next_start > start and overlap_length < overlap:
            next_start -= 1
            overlap_length += len(words[next_start]) + 1
        start = next_start

    return chunks
