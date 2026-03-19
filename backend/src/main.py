from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import text
from src.database import engine, init_db
from src.ingestion.handler import save_upload, load_dataset
from src.llm.analyst import chat, clear_history
from src.rag.vectorstore import build_vectorstore, is_indexed
from src.visualization.advanced_charts import AdvancedVisualizer, auto_visualize
from src.eda.advanced_summary import get_comprehensive_summary

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


@app.post("/visualize")
def visualize_data(dataset_id: int, question: str):
    try:
        df, filename = load_dataset(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        result = auto_visualize(df, question)
        if result is None:
            # Return a helpful message instead of error
            return {
                "type": "info", 
                "chart": None, 
                "description": "Could not generate a suitable visualization for this question. Try asking for 'correlation matrix', 'distribution of [column]', or 'scatter plot of [col1] vs [col2]'."
            }
        
        return result
        
    except Exception as e:
        print(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")


@app.get("/comprehensive-analysis/{dataset_id}")
def get_comprehensive_analysis(dataset_id: int):
    """Get comprehensive statistical analysis and visualizations"""
    try:
        df, filename = load_dataset(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get comprehensive statistical summary
        summary = get_comprehensive_summary(df)
        
        # Generate comprehensive visualizations
        visualizer = AdvancedVisualizer(df)
        charts = visualizer.create_comprehensive_dashboard()
        
        return {
            "filename": filename,
            "summary": summary,
            "visualizations": charts,
            "message": "Comprehensive analysis completed successfully"
        }
        
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error in comprehensive analysis: {str(e)}")


@app.post("/advanced-visualize")
def advanced_visualize(dataset_id: int, chart_type: str, columns: list = None, question: str = ""):
    """Create advanced visualizations with multiple chart types"""
    try:
        df, filename = load_dataset(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        visualizer = AdvancedVisualizer(df)
        chart = visualizer.create_custom_visualization(chart_type, columns, question)
        
        if chart:
            return {
                "type": chart_type,
                "chart": chart,
                "description": f"Advanced {chart_type} visualization",
                "columns_used": columns or []
            }
        else:
            raise HTTPException(status_code=400, detail="Could not generate the requested visualization")
            
    except Exception as e:
        print(f"Advanced visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating advanced visualization: {str(e)}")
