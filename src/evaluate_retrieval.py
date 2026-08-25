"""
Retrieval evaluation — проверяем, насколько хорошо работает поиск,
ДО того как результаты попадут в LLM для генерации ответа.

Важно разделять:
  - retrieval quality (нашли ли правильный chunk?)
  - generation quality (сгенерировала ли LLM хороший ответ на его основе?)
Если retrieval плохой — LLM не поможет, ответ будет неверным
независимо от качества промпта.

Метрика: hit_rate@k — доля запросов, где правильный документ
попал в top-k результатов поиска.
"""

from vector_store import vector_search
from hybrid_search import hybrid_search


# тестовый набор: запрос -> ожидаемый title документа, где есть ответ
# в реальном проекте это собирается вручную или через LLM-разметку
EVAL_QUERIES = [
    {"query": "who created python programming language", "expected_title": "Python"},
    {"query": "machine learning libraries in python", "expected_title": "Python"},
    # добавляй сюда свои пары после загрузки реальных документов
]


def hit_rate_at_k(search_fn, eval_queries: list[dict], k: int = 5) -> float:
    """
    Считает hit_rate@k для заданной функции поиска.

    search_fn — любая функция поиска (vector_search, hybrid_search),
    которая принимает query и top_k, возвращает список результатов с title.
    """
    hits = 0

    for item in eval_queries:
        results = search_fn(item["query"], top_k=k)
        found_titles = {r["title"] for r in results}

        if item["expected_title"] in found_titles:
            hits += 1

    return hits / len(eval_queries) if eval_queries else 0.0


def compare_search_methods(eval_queries: list[dict] = None, k: int = 5) -> dict:
    """
    Сравнивает hit_rate@k для vector search vs hybrid search.
    Это и есть финальная метрика проекта — какой метод поиска лучше.
    """
    queries = eval_queries or EVAL_QUERIES

    vector_hit_rate = hit_rate_at_k(
        lambda q, top_k: vector_search(q, top_k=top_k),
        queries, k=k,
    )
    hybrid_hit_rate = hit_rate_at_k(
        lambda q, top_k: hybrid_search(q, top_k=top_k, alpha=0.5),
        queries, k=k,
    )

    return {
        "vector_search_hit_rate": round(vector_hit_rate, 3),
        "hybrid_search_hit_rate": round(hybrid_hit_rate, 3),
        "k": k,
        "num_queries": len(queries),
    }


if __name__ == "__main__":
    report = compare_search_methods(k=5)

    print("=== Retrieval Evaluation ===\n")
    print(f"Vector search hit_rate@{report['k']}:  {report['vector_search_hit_rate']*100:.1f}%")
    print(f"Hybrid search hit_rate@{report['k']}:  {report['hybrid_search_hit_rate']*100:.1f}%")
    print(f"\nНа {report['num_queries']} тестовых запросах")