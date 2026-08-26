"""
main.py — точка входа проекта rag-hybrid-search.

Запускать из корня проекта:
    python main.py

Что делает:
    1. Загружает переменные окружения из .env
    2. Настраивает базу данных (таблица chunks + индекс)
    3. Загружает документы из wikipedia
    4. Делит их на chunks
    5. Считает эмбеддинги и сохраняет в pgvector
    6. Прогоняет тестовый hybrid_search запрос
    7. Показывает, что search_documents (инструмент агента) работает

Все реальные функции лежат в src/ — этот файл только вызывает их
в правильном порядке.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# .env должен быть загружен ДО импорта модулей из src,
# потому что vector_store.py читает os.getenv() на уровне модуля
load_dotenv()

# добавляем src/ в путь поиска модулей, чтобы работали
# внутренние импорты вида "from hybrid_search import hybrid_search"
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data import load_documents
from chunking import chunk_document
from vector_store import setup_database, insert_chunks
from hybrid_search import hybrid_search
from agent_tool import search_documents


def check_env():
    """Проверяет, что обязательные переменные окружения заданы."""
    required = ["OPENAI_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        print("Ошибка: не заданы переменные окружения:")
        for key in missing:
            print(f"  - {key}")
        print("\nСкопируй .env.example в .env и впиши свои значения.")
        sys.exit(1)


def run_pipeline(n_docs: int = 20):
    """Полный пайплайн: от загрузки документов до готового поиска."""

    print("Шаг 1/5: проверка переменных окружения...")
    check_env()

    print("Шаг 2/5: настройка базы данных...")
    setup_database()

    print(f"Шаг 3/5: загрузка {n_docs} документов из wikipedia...")
    documents = load_documents(n=n_docs)
    print(f"  Загружено документов: {len(documents)}")

    print("Шаг 4/5: chunking + эмбеддинги + сохранение в pgvector...")
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, strategy="sentence_based"))
    print(f"  Получено chunks: {len(all_chunks)}")
    insert_chunks(all_chunks)

    print("Шаг 5/5: проверка hybrid_search...")
    test_query = "programming language for machine learning"
    results = hybrid_search(test_query, top_k=3, alpha=0.5)

    print(f"\nРезультаты поиска по запросу: '{test_query}'")
    for r in results:
        print(f"  [{r['final_score']:.3f}] {r['title']} (chunk {r['chunk_index']})")

    print("\nГотово. Инструмент search_documents доступен для агента:")
    print(search_documents.invoke({"query": test_query}))


if __name__ == "__main__":
    run_pipeline(n_docs=20)