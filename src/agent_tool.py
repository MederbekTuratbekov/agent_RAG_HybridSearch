"""
Подключаем RAG как ещё один инструмент к агенту из проекта 3.

Идея: агент из langgraph_memory умел calculator и search (заглушку).
Теперь search становится РЕАЛЬНЫМ — ищет по нашей базе документов
через hybrid_search, а не возвращает фиктивный текст.

Это финальная точка проекта 4: агент, который может отвечать
на вопросы по конкретным документам, а не только "из головы" LLM.
"""

from langchain_core.tools import tool
from hybrid_search import hybrid_search


@tool
def search_documents(query: str) -> str:
    """
    Ищет релевантную информацию в базе документов.
    Используй этот инструмент, когда нужно найти факты
    из загруженных документов, а не отвечать по общим знаниям.
    """
    results = hybrid_search(query, top_k=3, alpha=0.5)

    if not results:
        return "По этому запросу ничего не найдено в базе документов."

    formatted = []
    for r in results:
        formatted.append(f"[Источник: {r['title']}]\n{r['text']}")

    return "\n\n---\n\n".join(formatted)


def build_full_pipeline():
    """
    Полный пайплайн проекта 4:
        1. Загрузка документов
        2. Chunking
        3. Эмбеддинги + сохранение в pgvector
        4. Проверка через vector_search / hybrid_search
        5. Готовый инструмент search_documents для агента

    Запускай по шагам через отдельные файлы (data.py, chunking.py,
    vector_store.py) — этот файл просто показывает итоговую сборку.
    """
    from data import load_documents
    from chunking import chunk_document
    from vector_store import setup_database, insert_chunks

    print("Шаг 1: настройка базы данных...")
    setup_database()

    print("Шаг 2: загрузка документов...")
    documents = load_documents(n=20)

    print("Шаг 3: chunking...")
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, strategy="sentence_based"))
    print(f"Получено {len(all_chunks)} chunks из {len(documents)} документов")

    print("Шаг 4: эмбеддинги + сохранение в pgvector...")
    insert_chunks(all_chunks)

    print("\nГотово. Инструмент search_documents готов к использованию агентом.")


if __name__ == "__main__":
    build_full_pipeline()