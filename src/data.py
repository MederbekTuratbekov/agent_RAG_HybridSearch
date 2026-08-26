"""
Загрузка датасета документов для RAG-системы.

Вместо случайной выборки загружаем конкретные статьи по списку
заголовков — так тестовые запросы гарантированно находят
релевантные документы, и легко проверить, что hybrid search
работает правильно.

Используем wikimedia/wikipedia (simple english) — Parquet-формат,
без loading-скриптов.

Установка:
    pip install datasets
"""

from datasets import load_dataset


# заголовки статей, которые точно должны попасть в базу.
# добавляй сюда любые темы, которые хочешь протестировать поиском
DEFAULT_TITLES = [
    "Python (programming language)",
    "Machine learning",
    "R (programming language)",
    "Java (programming language)",
    "JavaScript",
    "Artificial intelligence",
    "Deep learning",
    "Neural network",
    "Data science",
    "Algorithm",
    "Programming language",
    "Computer science",
    "Natural language processing",
    "Statistics",
    "Database",
]


def load_documents_by_titles(titles: list[str] = None) -> list[dict]:
    """
    Загружает статьи по точному списку заголовков из wikimedia/wikipedia.

    Проходит по всему датасету один раз и отбирает только строки,
    чей title совпадает с одним из titles. Первый запуск медленнее
    (полный проход по датасету), но результат кэшируется HF datasets.

    Возвращает список:
        {"id": str, "title": str, "text": str}
    """
    titles = titles or DEFAULT_TITLES
    titles_set = set(titles)

    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    filtered = dataset.filter(lambda row: row["title"] in titles_set)

    documents = []
    for row in filtered:
        documents.append({
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
        })

    found_titles = {d["title"] for d in documents}
    missing = titles_set - found_titles
    if missing:
        print(f"  Не найдены в датасете: {sorted(missing)}")

    return documents


def load_documents(n: int = 50, seed: int = 42) -> list[dict]:
    """
    Оставлено для обратной совместимости — случайная выборка n статей.
    Для целевого тестирования используй load_documents_by_titles().
    """
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
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