from fastapi import FastAPI, UploadFile, File
from sqlalchemy import text
from src.database import engine
import shutil
import os


app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/test-db")
def test_db():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        return {"status": "connected", "result": [row[0] for row in result]}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO datasets (name) VALUES (:name)"),
            {"name": file.filename}
        )
        conn.commit()

    return {"message": "File uploaded successfully"}

