from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import text
from src.database import engine, init_db
from src.ingestion.handler import save_upload, load_dataset
from src.llm.analyst import chat, clear_history
from src.rag.vectorstore import build_vectorstore, is_indexed

app = FastAPI()


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/test-db")
def test_db():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        return {"status": "connected", "result": [row[0] for row in result]}


@app.post("/upload")
def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    return save_upload(file, background_tasks)


@app.get("/index-status/{dataset_id}")
def index_status(dataset_id: int):
    return {"indexed": is_indexed(dataset_id)}


def rebuild_index(dataset_id: int):
    df, filename = load_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    build_vectorstore(dataset_id, df)
    return {"message": f"Index rebuilt for dataset {dataset_id} ({filename})"}


@app.post("/chat")
def chat_with_dataset(dataset_id: int, question: str):
    df, filename = load_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return chat(dataset_id, question, df)


@app.delete("/chat/{dataset_id}")
def reset_chat(dataset_id: int):
    clear_history(dataset_id)
    return {"message": "Conversation cleared"}
