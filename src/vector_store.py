"""
Эмбеддинги + pgvector — векторное хранилище на твоём PostgreSQL.

pgvector — расширение PostgreSQL для хранения векторов
и быстрого поиска по косинусному расстоянию. Раз PostgreSQL
уже в твоём стеке, не нужно поднимать отдельную векторную БД.

Установка PostgreSQL-расширения (один раз):
    CREATE EXTENSION IF NOT EXISTS vector;

Установка Python-пакетов:
    pip install openai psycopg2-binary pgvector python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

# load_dotenv() здесь, а не только в main.py — чтобы файл работал
# и при прямом запуске (python src/vector_store.py), а не только
# как импортируемый модуль внутри пайплайна
load_dotenv()

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 измерений, дешёвая и качественная

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DBNAME", "rag_demo"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}


def get_connection():
    """Открывает соединение с PostgreSQL и регистрирует тип vector."""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def setup_database():
    """
    Создаёт таблицу chunks с векторной колонкой, если её ещё нет.

    embedding vector(1536) — 1536, потому что именно такую размерность
    выдаёт text-embedding-3-small. Если сменишь модель эмбеддингов,
    размерность нужно поменять здесь тоже.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT,
            title TEXT,
            chunk_index INT,
            text TEXT,
            embedding vector(1536)
        );
    """)
    # индекс для быстрого поиска по косинусному расстоянию
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("База данных готова: таблица chunks + индекс созданы")


def get_embedding(text: str) -> list[float]:
    """Получает эмбеддинг для одного текста через OpenAI API."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def insert_chunks(chunks: list[dict]):
    """
    Считает эмбеддинги для списка chunks и сохраняет их в PostgreSQL.

    Батчим по одному ради простоты — для реального проекта
    стоит использовать batch embeddings API, чтобы не делать
    сотни отдельных HTTP-вызовов.
    """
    conn = get_connection()
    cur = conn.cursor()

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])

        cur.execute(
            """
            INSERT INTO chunks (doc_id, title, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (chunk["doc_id"], chunk["title"], chunk["chunk_index"], chunk["text"], embedding),
        )

        if (i + 1) % 10 == 0:
            print(f"Обработано {i + 1}/{len(chunks)} chunks")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Сохранено {len(chunks)} chunks в базу")


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Векторный поиск: находит top_k наиболее похожих chunks на query
    по косинусному расстоянию.

    Оператор <=> в pgvector — это косинусное расстояние
    (чем меньше значение, тем более похожи вектора).
    """
    query_embedding = get_embedding(query)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT doc_id, title, chunk_index, text, embedding <=> %s AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s
        """,
        (query_embedding, top_k),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "doc_id": row[0],
            "title": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "distance": float(row[4]),
        }
        for row in rows
    ]
