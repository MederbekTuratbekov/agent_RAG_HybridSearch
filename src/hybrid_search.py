"""
Hybrid search = BM25 (keyword-поиск) + vector search (семантический поиск).

Зачем комбинировать:
  - vector search находит "похожее по смыслу", но иногда промахивается
    на точных терминах (имена, коды, редкие слова)
  - BM25 находит "точное совпадение слов", но не понимает синонимы
    и перефразировки
  - вместе они компенсируют слабости друг друга

Установка:
    pip install rank-bm25
"""

from rank_bm25 import BM25Okapi

from vector_store import vector_search, get_connection


def load_all_chunks_for_bm25() -> tuple[BM25Okapi, list[dict]]:
    """
    Загружает все chunks из PostgreSQL и строит BM25-индекс поверх них.

    BM25Okapi работает с токенизированными текстами (список слов),
    поэтому просто разбиваем текст по пробелам — для демонстрации
    этого достаточно, в проде стоит использовать нормальный токенизатор.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT doc_id, title, chunk_index, text FROM chunks;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    chunks = [
        {"doc_id": row[0], "title": row[1], "chunk_index": row[2], "text": row[3]}
        for row in rows
    ]

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    return bm25, chunks


def bm25_search(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Keyword-поиск через BM25 — возвращает top_k наиболее релевантных chunks."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [
        {**chunk, "bm25_score": float(score)}
        for chunk, score in scored_chunks[:top_k]
    ]


def normalize_scores(results: list[dict], score_key: str) -> list[dict]:
    """
    Нормализует scores в диапазон [0, 1] через min-max normalization.
    Нужно, чтобы честно комбинировать BM25-score и vector-distance —
    у них изначально разные шкалы значений.
    """
    if not results:
        return results

    scores = [r[score_key] for r in results]
    min_score, max_score = min(scores), max(scores)
    range_score = max_score - min_score if max_score != min_score else 1

    for r in results:
        r[f"{score_key}_normalized"] = (r[score_key] - min_score) / range_score

    return results


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
    """
    Комбинирует vector search и BM25 через взвешенную сумму.

    alpha — вес vector search (1 - alpha — вес BM25).
    alpha=0.5 — равный вклад обоих методов.
    alpha=1.0 — только векторный поиск.
    alpha=0.0 — только BM25.
    """
    vector_results = vector_search(query, top_k=top_k * 2)  # берём с запасом
    bm25_instance, all_chunks = load_all_chunks_for_bm25()
    bm25_results = bm25_search(query, bm25_instance, all_chunks, top_k=top_k * 2)

    # vector distance: меньше = лучше, поэтому инвертируем в score (больше = лучше)
    for r in vector_results:
        r["vector_score"] = 1 - r["distance"]

    vector_results = normalize_scores(vector_results, "vector_score")
    bm25_results = normalize_scores(bm25_results, "bm25_score")

    # объединяем по уникальному ключу (doc_id + chunk_index)
    combined: dict[str, dict] = {}

    for r in vector_results:
        key = f"{r['doc_id']}_{r['chunk_index']}"
        combined[key] = {**r, "final_score": alpha * r.get("vector_score_normalized", 0)}

    for r in bm25_results:
        key = f"{r['doc_id']}_{r['chunk_index']}"
        bm25_contribution = (1 - alpha) * r.get("bm25_score_normalized", 0)
        if key in combined:
            combined[key]["final_score"] += bm25_contribution
        else:
            combined[key] = {**r, "final_score": bm25_contribution}

    sorted_results = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
    return sorted_results[:top_k]


if __name__ == "__main__":
    query = "programming language for machine learning"
    results = hybrid_search(query, top_k=3, alpha=0.5)

    print(f"=== Hybrid search: '{query}' ===\n")
    for r in results:
        print(f"[{r['final_score']:.3f}] {r['title']} (chunk {r['chunk_index']})")
        print(f"  {r['text'][:100]}...")
        print()