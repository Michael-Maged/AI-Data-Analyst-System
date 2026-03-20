# AI-Data-Analyst-System

You are building a full AI Data Analyst System, and at this stage you have already completed the core infrastructure layer: a Dockerized FastAPI backend connected to a PostgreSQL database, with a persistent volume-mounted uploads system that allows CSV files to be stored reliably even if containers restart; your backend successfully handles file ingestion, saves dataset metadata in the database, reads datasets using pandas, and returns basic structural insights such as columns and row counts, which means your data ingestion pipeline is fully functional. The next objective is to evolve this into a complete intelligent system by layering advanced capabilities on top of this foundation. You will implement a robust ingestion module in src/ingestion to standardize dataset handling, followed by a preprocessing layer in src/preprocessing responsible for cleaning data, handling missing values, type inference, and normalization, ensuring that all datasets are analysis-ready. On top of that, you will build an exploratory data analysis (EDA) module in src/eda to automatically compute descriptive statistics, correlations, distributions, and data summaries, enabling both automated insights and structured metadata extraction. The core of the system will be the LLM integration in src/llm, where models like Qwen Coder, Phi-3, or similar will be used to translate natural language questions into executable pandas code; this layer will be engineered with production-grade practices including structured JSON outputs, schema-aware prompting (injecting columns, data types, and sample rows), retry and self-healing mechanisms for failed executions, and strict sandboxing to prevent unsafe code execution. Alongside this, you will develop a models layer (src/models) to define data schemas, API contracts, and possibly future ML models for predictive analytics. For user-facing insights, a visualization module (src/visualization) will generate dynamic plots (matplotlib/plotly) based on user queries, allowing the system to return not just numerical answers but also charts and graphs. To further enhance usability, you will integrate a voice interface: speech-to-text using Whisper for capturing user queries and text-to-speech (e.g., pyttsx3 or a higher-quality API) for delivering spoken responses, effectively transforming the system into a conversational AI analyst. On the frontend/GUI side, you will build an interactive interface (likely in React) that supports file uploads, dataset browsing, chat-based querying, and visualization rendering, creating a seamless user experience. Finally, you will refine the system architecture by modularizing your current monolithic main.py into the appropriate folders (ingestion, preprocessing, eda, llm, visualization), implement logging and error handling, and optionally extend the system with advanced features such as storing datasets as SQL tables for hybrid querying, maintaining conversational memory, and supporting multi-dataset analysis. The end result will be a fully integrated platform that allows users to upload data, explore it, query it in natural language, visualize results, and interact via both text and voice — effectively delivering a complete, production-ready AI-powered data analysis environment.

---

## Architecture

```
AI-Data-Analyst-System/
├── backend/
│   ├── src/
│   │   ├── main.py              # FastAPI app + all endpoints
│   │   ├── database.py          # SQLAlchemy engine + init_db()
│   │   ├── ingestion/
│   │   │   └── handler.py       # save_upload(), load_dataset(), _read_file()
│   │   ├── eda/
│   │   │   └── summary.py       # get_dataset_summary(df)
│   │   ├── rag/
│   │   │   └── vectorstore.py   # build_vectorstore(), ChromaDB + nomic-embed-text
│   │   └── llm/
│   │       └── analyst.py       # ConversationalRetrievalChain, self-healing code exec
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── app.py                   # Streamlit UI: upload, chat, data preview
├── data/
│   ├── chroma/                  # ChromaDB vector stores (gitignored)
│   ├── processed/               # Cleaned datasets (gitignored)
│   └── raw/                     # Raw data (gitignored)
├── docker-compose.yml
└── .gitignore
```

---

## Key Design Decisions

- **Ollama on host**: All LLM/embedding calls use `http://host.docker.internal:11434` since Ollama runs on the host, not in Docker
- **Background indexing**: `build_vectorstore` runs as a FastAPI `BackgroundTask` so uploads return immediately; frontend polls `/index-status/{id}`
- **Two-mode LLM**: Analyst responds with `mode: code` (executes pandas code) or `mode: analysis` (direct text answer) based on question type
- **Self-healing**: On code execution failure, the error is fed back to the LLM for one retry
- **Row chunking**: ChromaDB documents use 10-row chunks with 1500-char truncation to avoid context length errors
- **ChromaDB telemetry**: Suppressed via `ANONYMIZED_TELEMETRY=False` env var

---

## Current State

- ✅ Dockerized FastAPI backend + PostgreSQL
- ✅ CSV and Excel file upload with persistent volume
- ✅ Dataset metadata stored in PostgreSQL
- ✅ Advanced data preprocessing with cleaning and normalization
- ✅ RAG pipeline with LangChain + ChromaDB + nomic-embed-text
- ✅ Conversational chat with custom memory system
- ✅ LLM integration with qwen2.5-coder via Ollama
- ✅ Two-mode analyst: code execution + direct analysis
- ✅ Self-healing code generation (retry on failure)
- ✅ Professional visualization suite with 8+ chart types
- ✅ Comprehensive statistical analysis with advanced metrics
- ✅ Data quality assessment and recommendations
- ✅ Streamlit frontend with comprehensive analysis dashboard
- ✅ Background vector store indexing on upload

---

## Roadmap

### Phase 1 — Fix & Stabilize ✅ COMPLETED
- [X] Test full flow: upload → index → chat
- [X] Verify RAG is retrieving context correctly
- [X] Fix all remaining bugs

### Phase 2 — Preprocessing Layer ✅ COMPLETED
- [X] Handle missing values (fill, drop, flag)
- [X] Type inference and correction
- [X] Normalization and standardization
- [X] Save cleaned data to `data/processed/`
- [X] RAG indexes cleaned data instead of raw

### Phase 3 — Advanced Visualizations ✅ COMPLETED
- [X] Professional visualization suite with 8+ chart types
- [X] Comprehensive statistical analysis dashboard
- [X] Data quality assessment and recommendations
- [X] Auto-visualization based on natural language queries
- [X] Advanced EDA with correlation analysis and outlier detection

### Phase 4 — Smarter LLM Layer ← NEXT
- [ ] Streaming responses (word by word like ChatGPT)
- [ ] Better prompt engineering for complex analytical questions
- [ ] Multi-turn reasoning referencing previous answers

### Phase 5 — React Frontend
- [ ] Migrate from Streamlit to React
- [ ] Proper chat UI with streaming
- [ ] Drag and drop file upload
- [ ] Interactive charts with Plotly

### Phase 6 — Voice Interface
- [ ] Whisper for speech-to-text
- [ ] pyttsx3 or ElevenLabs for text-to-speech
- [ ] Mic button in the UI

### Phase 7 — Migrate to Supabase
- [ ] Replace local PostgreSQL container with Supabase hosted database
- [ ] Update DATABASE_URL in .env to point to Supabase connection string
- [ ] Test all existing DB operations (upload metadata, dataset lookup)
- [ ] Store ChromaDB embeddings in Supabase pgvector instead of local Chroma (optional)

### Phase 8 — Notebooks
- [ ] Evaluate if notebooks are needed for experimentation or prototyping
- [ ] If yes: use `notebooks/exploration.ipynb` for EDA experiments before productionizing
- [ ] Document findings and move stable code into `src/` modules
