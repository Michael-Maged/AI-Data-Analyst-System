import os
import shutil
import pandas as pd
from sqlalchemy import text
from src.database import engine

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def save_upload(file, background_tasks) -> dict:
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = _read_file(file_path)

    with engine.connect() as conn:
        result = conn.execute(
            text("INSERT INTO datasets (name) VALUES (:name) RETURNING id"),
            {"name": file.filename}
        ).fetchone()
        conn.commit()
        dataset_id = result[0]

    # Preprocess and build RAG vector store in background
    from src.rag.vectorstore import build_vectorstore
    from src.preprocessing.cleaner import clean
    cleaned_df = clean(df, dataset_id)
    background_tasks.add_task(build_vectorstore, dataset_id, cleaned_df)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "columns": list(df.columns),
        "rows_count": df.shape[0],
        "columns_count": df.shape[1],
        "preview": df.head().to_dict(orient="records"),
        "message": "File uploaded. Index is being built in the background."
    }


def load_dataset(dataset_id: int) -> tuple[pd.DataFrame, str]:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT name FROM datasets WHERE id = :id"),
            {"id": dataset_id}
        ).fetchone()

    if not result:
        return None, None

    # Try to load cleaned data first, fallback to raw
    cleaned_path = f"data/processed/{dataset_id}.csv"
    if os.path.exists(cleaned_path):
        return pd.read_csv(cleaned_path), result[0]
    
    file_path = f"{UPLOAD_DIR}/{result[0]}"
    return _read_file(file_path), result[0]
