"""
Chunking — разбивка длинных документов на маленькие куски (chunks).

Зачем нужен chunking:
  - эмбеддинг-модели имеют лимит на длину входа
  - маленькие chunks дают более точный поиск (не "вся статья",
    а конкретный абзац, где реально есть ответ)
  - слишком маленькие chunks теряют контекст — баланс важен

Здесь две стратегии для сравнения:
  1. fixed_size — режем по количеству символов с overlap
  2. sentence_based — режем по предложениям, группируя в chunks
"""

import re


def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Разбивка по фиксированному числу символов с overlap (перекрытием).

    overlap нужен, чтобы не терять контекст на границе chunks —
    если важное предложение попало ровно на стык, у нас есть шанс
    поймать его целиком в соседнем chunk.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def chunk_sentence_based(text: str, max_chunk_size: int = 500) -> list[str]:
    """
    Разбивка по предложениям — группируем предложения в chunk,
    пока не превысим max_chunk_size, потом начинаем новый chunk.

    Плюс перед fixed_size: не режем предложение посередине,
    смысл каждого chunk остаётся целостным.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_document(doc: dict, strategy: str = "sentence_based") -> list[dict]:
    """
    Разбивает один документ на chunks и добавляет метаданные
    (title, doc_id, chunk_index) — пригодится для отображения
    источника при выдаче ответа.
    """
    if strategy == "fixed_size":
        raw_chunks = chunk_fixed_size(doc["text"])
    else:
        raw_chunks = chunk_sentence_based(doc["text"])

    return [
        {
            "doc_id": doc["id"],
            "title": doc["title"],
            "chunk_index": i,
            "text": chunk_text,
        }
        for i, chunk_text in enumerate(raw_chunks)
        if len(chunk_text) > 20  # выкидываем совсем короткие мусорные chunks
    ]
