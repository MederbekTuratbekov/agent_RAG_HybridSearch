"""
Загрузка датасета документов для RAG-системы.

Используем wikipedia subset — компактную выборку статей.
Полная wikipedia на HF огромная, поэтому берём wikipedia
simple english (упрощённая версия) — быстрее скачивается,
достаточно для демонстрации RAG-пайплайна.

Установка:
    pip install datasets
"""

from datasets import load_dataset


def load_documents(n: int = 50, seed: int = 42) -> list[dict]:
    """
    Загружает n статей из wikipedia (simple english subset).

    Возвращает список:
        {"id": str, "title": str, "text": str}
    """
    dataset = load_dataset("wikipedia", "20220301.simple", split="train", trust_remote_code=True)
    shuffled = dataset.shuffle(seed=seed)
    sample = shuffled.select(range(n))

    documents = []
    for row in sample:
        documents.append({
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
        })

    return documents
