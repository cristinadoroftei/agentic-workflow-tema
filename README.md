# Document Analyst Agent with RAG

A conversational agent that extracts structured data from documents (invoices, contracts), stores them in PostgreSQL with pgvector embeddings, and answers questions using semantic search (RAG).

Built as homework for Skillab AI course (Lessons 1-4).

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           Extraction Pipeline             │
                    │  load → classify → chunk → extract → DB  │
                    └──────────────┬───────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
     extracted_data/        documents table       document_chunks table
     (JSON files)          (JSONB metadata)       (pgvector embeddings)
                                                         │
                                                         ▼
                    ┌──────────────────────────────────────────┐
                    │              ReAct Agent                  │
                    │  user question → search_documents tool    │
                    │  → RAG context → LLM answer              │
                    └──────────────────────────────────────────┘
```

## Project Structure

```
proiect/
├── agent.py                  # ReAct agent — Claude native format, Gemini fallback
├── pipeline/
│   ├── loader.py             # File loader registry (PDF, DOCX, TXT, CSV)
│   ├── schemas.py            # Invoice + Contract Pydantic models
│   └── pipeline.py           # Full pipeline: load → classify → chunk → extract → DB + RAG
├── db/
│   ├── database.py           # SQLAlchemy engine, SessionLocal, transaction()
│   ├── models.py             # Document (JSONB metadata) + DocumentChunk (Vector(384))
│   └── repositories/
│       ├── document_repository.py   # CRUD + filter_by_metadata
│       └── chunk_repository.py      # CRUD + similarity_search (cosine distance)
├── rag/
│   └── rag_service.py        # Embed, search, get_context (all-MiniLM-L6-v2)
├── tools/
│   ├── registry.py           # TOOL_REGISTRY + @register_tool decorator
│   ├── params_models.py      # Pydantic params for each tool
│   ├── basic_tools.py        # calculator, get_datetime, web_search, search_documents
│   └── tool_wrapper.py       # ToolWrapper.call() + catalog()
├── prompts/
│   ├── registry.py           # PromptRegistry — loads YAML + renders with Jinja2
│   ├── planner.yaml          # Planning mode prompt
│   ├── analyst.yaml          # Analysis mode prompt
│   ├── summary.yaml          # Summarization mode prompt
│   └── extract.yaml          # Extraction mode prompt
├── alembic/                  # Database migrations
│   └── versions/
│       ├── dcbab4b1d6b8_...  # Initial: CREATE EXTENSION vector, documents, document_chunks
│       └── 3579f0e1ebe7_...  # HNSW index on embedding column
├── verify_db.py              # DB verification script (documents, chunks, hybrid search test)
├── docker-compose.yml        # pgvector/pgvector:pg16 on port 5434
├── samples/                  # Test documents (2 invoices, 2 contracts)
├── extracted_data/           # JSON output from extraction pipeline
├── .env                      # API keys + DB credentials (not committed)
└── .env.example
```

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic google-genai langchain-text-splitters sentence-transformers \
            sqlalchemy alembic pgvector psycopg2-binary pydantic \
            httpx pyyaml jinja2 python-dotenv
```

### 2. Environment variables

```bash
cp .env.example .env
```

Add to `.env`:
```
ANTHROPIC_API_KEY=sk-...
GEMINI_API_KEY=...

POSTGRES_USER=demo
POSTGRES_PASSWORD=demo123
POSTGRES_DB=rag_demo
```

### 3. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

### 4. Run database migrations

```bash
alembic upgrade head
```

This creates the `vector` extension, `documents` table, `document_chunks` table, and HNSW index.

### 5. Run the extraction pipeline

```bash
python3 -c "
from pipeline.pipeline import process
from pathlib import Path

for f in sorted(Path('samples').iterdir()):
    print(f'\n=== Processing: {f.name} ===')
    process(f)
"
```

This processes each sample file through the full pipeline:
1. Load the document
2. Classify (invoice vs contract) using LLM
3. Chunk if needed (contracts use first 3 chunks for extraction)
4. Extract structured data into Pydantic models
5. Save as JSON to `extracted_data/`
6. Save to PostgreSQL (document + JSONB metadata)
7. Embed chunks (500 char, all-MiniLM-L6-v2) and store in `document_chunks`

### 6. Verify the database

```bash
python3 verify_db.py
```

This prints a summary of documents, chunks, metadata, and runs a hybrid search test.

To explore manually with psql:
```bash
docker compose exec db psql -U demo -d rag_demo
```
```sql
SELECT id, filename, metadata->>'doc_type' AS type FROM documents;
SELECT document_id, count(*) FROM document_chunks GROUP BY document_id;
```

### 7. Run the agent

```bash
python3 agent.py
```

Ask questions about your documents:
```
> Cine este furnizorul de pe factura 001?
> Care sunt partile contractului de consultanta?
> Ce suma totala apare pe factura 002?
```

## Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_documents` | Semantic search over stored documents (invoices, contracts) via RAG | `query: str`, `top_k: int` (default 3) |
| `calculator` | Evaluates math expressions | `expression: str` |
| `get_datetime` | Returns current date/time in a timezone | `timezone: str` (default UTC) |
| `web_search` | Searches the web via DuckDuckGo | `query: str`, `max_results: int` |

## LLM Providers

Both the agent and the pipeline use Claude as default with Gemini fallback:

| Component | Default | Fallback |
|-----------|---------|----------|
| Agent | Claude Haiku 4.5 (native tool use) | Gemini 2.5 Flash (OpenAI format) |
| Pipeline | Claude Haiku 4.5 (fake tool trick) | Gemini 2.0 Flash (response_schema) |

## Agent Modes

Switch modes at runtime with `/mode <name>`:

| Mode | Purpose |
|------|---------|
| `planner` | Breaks complex questions into steps |
| `analyst` | Researches and provides detailed analysis |
| `summary` | Returns short, concise answers |
| `extract` | Pulls specific data points |

Commands: `/mode <name>`, `/modes`, `exit`
