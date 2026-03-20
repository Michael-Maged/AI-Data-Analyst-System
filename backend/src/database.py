from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/ai_analyst")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                rows_count INTEGER,
                columns_count INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # Add columns if they don't exist (for existing DBs)
        for col, coltype in [("rows_count", "INTEGER"), ("columns_count", "INTEGER")]:
            try:
                conn.execute(text(f"ALTER TABLE datasets ADD COLUMN IF NOT EXISTS {col} {coltype}"))
            except Exception:
                pass
        conn.commit()