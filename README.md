# RAG Hybrid Search

RAG-система с **гибридным поиском** (BM25 + vector search) на своём стеке — PostgreSQL/`pgvector` вместо отдельной векторной БД. Финальная точка: реальный `search_documents` tool, подключаемый к [LangGraph-агенту](https://github.com/MederbekTuratbekov/agent_LanggraphAgentMemory.git) вместо заглушки `search`.

## Место в линейке проектов

Это четвёртый проект в серии от простого промптинга к полноценному агенту:

1. Промптинг-техники на SQuAD
2. Ручной ReAct-агент с калькулятором
3. Тот же агент на LangGraph + память (short-term/long-term)
4. **→ этот проект** — `search` в агенте перестаёт быть заглушкой и начинает реально искать по базе документов

## Зачем гибридный поиск, а не только vector search

| Метод | Сильная сторона | Слабая сторона |
|---|---|---|
| Vector search | понимает смысл, синонимы, перефразировки | промахивается на точных терминах (имена, коды, редкие слова) |
| BM25 (keyword) | точное совпадение слов | не понимает синонимы и смысл |
| **Hybrid (оба)** | компенсируют слабости друг друга | требует нормализации разных шкал score |

Комбинация — взвешенная сумма нормализованных score:

```
final_score = alpha * vector_score + (1 - alpha) * bm25_score
```

`alpha=0.5` — равный вклад; `alpha=1.0` — чистый vector search; `alpha=0.0` — чистый BM25. Оба score приводятся к `[0, 1]` через min-max нормализацию перед суммированием — без этого шкалы несравнимы (косинусное расстояние и BM25-score живут в разных диапазонах).

## Pipeline

```
Wikipedia (по списку заголовков)
        │
        ▼
   Chunking (sentence-based / fixed-size)
        │
        ▼
   Эмбеддинги (text-embedding-3-small)
        │
        ▼
   PostgreSQL + pgvector
        │
        ├──► vector_search (косинусное расстояние)
        ├──► BM25 (keyword)
        │
        ▼
   hybrid_search (взвешенная сумма)
        │
        ▼
   search_documents (LangChain tool для агента)
```

## Chunking: две стратегии

- **`fixed_size`** — режет по количеству символов с overlap (перекрытием), чтобы не терять контекст на стыке chunks
- **`sentence_based`** (используется по умолчанию) — группирует целые предложения в chunk, не разрезая их посередине — сохраняет смысловую целостность

Оба варианта в `chunking.py` для прямого сравнения на своих данных.

## Оценка качества поиска

Ключевая идея: **retrieval quality ≠ generation quality**. Если поиск находит не тот документ — LLM не спасёт ответ, каким бы хорошим ни был промпт. Поэтому качество поиска проверяется отдельно, до генерации.

Метрика — `hit_rate@k`: доля запросов, где ожидаемый документ попал в top-k результатов.

```python
compare_search_methods()
# {
#   "vector_search_hit_rate": 0.72,
#   "hybrid_search_hit_rate": 0.86,
#   "k": 5,
#   "num_queries": 12
# }
```

Тестовый набор запросов с ожидаемыми документами — в `evaluate_retrieval.py` (`EVAL_QUERIES`), пополняется вручную под свои данные.

## Структура проекта

```
rag-hybrid-search/
├── src/
│   ├── main.py                  # полный пайплайн одной командой
│   ├── data.py                    # загрузка wikipedia по списку заголовков
│   ├── chunking.py                  # fixed_size + sentence_based
│   ├── vector_store.py                # pgvector: setup, insert, vector_search
│   ├── hybrid_search.py                 # BM25 + vector, взвешенное объединение
│   ├── evaluate_retrieval.py              # hit_rate@k, сравнение методов
│   └── agent_tool.py                        # search_documents tool для агента
├── requirements.txt
├── .env.example
└── .gitignore
```

## Установка

```bash
git clone https://github.com/<username>/rag-hybrid-search.git
cd rag-hybrid-search
pip install -r requirements.txt
```

Подними PostgreSQL с `pgvector` (проще всего через Docker):

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres ankane/pgvector
```

Создай `.env` на основе `.env.example`:

```
OPENAI_API_KEY=sk-...
PG_HOST=localhost
PG_PORT=5432
PG_DBNAME=rag_demo
PG_USER=postgres
PG_PASSWORD=postgres
```

## Запуск

```bash
python main.py
```

Полный пайплайн одной командой:
1. Проверка переменных окружения
2. Настройка БД (таблица `chunks` + `ivfflat`-индекс по косинусному расстоянию)
3. Загрузка ~15 статей Wikipedia (Python, Machine learning, Neural network и т.д.)
4. Chunking + эмбеддинги + сохранение в `pgvector`
5. Тестовый запрос через `vector_search` и `hybrid_search` с выводом обоих результатов для сравнения
6. Проверка готового `search_documents` tool

Пример вывода:

```
Результаты поиска по запросу: 'programming language for machine learning'
  [0.847] Python (programming language) (chunk 2)
  [0.612] Machine learning (chunk 0)
  [0.598] R (programming language) (chunk 1)
```

## Технологии

- **PostgreSQL + `pgvector`** — векторное хранилище на существующем стеке, без отдельной векторной БД
- **`rank-bm25`** — keyword-поиск (BM25Okapi)
- **OpenAI `text-embedding-3-small`** — эмбеддинги, 1536 измерений
- **LangChain `@tool`** — обёртка `search_documents` под агента
- **HuggingFace `datasets`** (`wikimedia/wikipedia`) — источник документов, Parquet-формат без loading-скриптов

## Возможные улучшения

- [ ] Batch embeddings API вместо отдельного HTTP-вызова на каждый chunk
- [ ] Reranking (cross-encoder) поверх top-k результатов hybrid search
- [ ] Автоматическая генерация `EVAL_QUERIES` через LLM-разметку вместо ручной
- [ ] Кэширование эмбеддингов запроса при повторных вызовах `hybrid_search`

## Лицензия

MIT