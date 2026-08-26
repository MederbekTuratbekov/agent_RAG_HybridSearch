# RAG Basics

Часть 4 из 5 в серии проектов по блоку `agent_junior`.

## Problem

Агент из проектов 2–3 отвечает только "из головы" LLM или через
заглушку поиска. Нужен реальный поиск по документам —
Retrieval-Augmented Generation.

## Approach

- Датасет: HuggingFace `wikipedia` (simple english subset, 20 статей)
- **Chunking**: две стратегии — `fixed_size` (с overlap) и
  `sentence_based` (по целым предложениям), сравнение в `chunking.py`
- **Эмбеддинги**: OpenAI `text-embedding-3-small` (1536 измерений)
- **Векторное хранилище**: **pgvector** на твоём PostgreSQL —
  не нужна отдельная векторная БД
- **Hybrid search**: комбинация BM25 (keyword) + vector search
  через взвешенную сумму нормализованных scores (`hybrid_search.py`)
- **Retrieval evaluation**: hit_rate@k — сравнение vector search
  против hybrid search на тестовых запросах
- Финал: `search_documents` — реальный RAG-инструмент,
  который встраивается в агента из проекта 3 вместо заглушки

## Results

`evaluate_retrieval.py` выводит hit_rate@5 для двух методов поиска —
это метрика для резюме ("RAG-система, hit_rate@5 = XX%").

## How to run

```bash
pip install -r requirements.txt

# PostgreSQL с pgvector (Docker):
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres ankane/pgvector

# Windows PowerShell:
$env:OPENAI_API_KEY="твой_ключ"
$env:PG_HOST="localhost"
$env:PG_PASSWORD="postgres"

python agent_tool.py             # полный пайплайн: данные -> chunks -> embeddings -> pgvector
python hybrid_search.py           # проверка hybrid search
python evaluate_retrieval.py      # метрики hit_rate
```

## Структура проекта

```
rag-hybrid-search/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── src/
    ├── main.py
    ├── data.py                 # загрузка wikipedia документов
    ├── chunking.py                # fixed_size и sentence_based стратегии
    ├── vector_store.py               # pgvector: setup, insert, vector_search
    ├── hybrid_search.py                 # BM25 + vector combined
    ├── evaluate_retrieval.py               # hit_rate@k метрика
    └── agent_tool.py                          # search_documents tool + полный пайплайн
```

## Связь с проектом 3

`search_documents` из этого проекта — прямая замена заглушки
`search_tool` из `langgraph_memory/tools.py`. Подставь эту функцию
в список TOOLS графа — и агент начнёт отвечать по реальным документам.