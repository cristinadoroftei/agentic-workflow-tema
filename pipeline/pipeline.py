import json
import os
from pathlib import Path
from enum import Enum
from typing import Type, TypeVar

import anthropic
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from pipeline.loader import load_document
from pipeline.schemas import Invoice, Contract
from db.database import transaction
from db.repositories import DocumentRepository
from rag.rag_service import RAGService

load_dotenv()

# --- Provider registry for structured output (Claude default, Gemini fallback) ---

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GEMINI_MODEL = "gemini-2.0-flash"
PROVIDER_ORDER = ["claude", "gemini"]

T = TypeVar("T", bound=BaseModel)


def _call_claude(prompt: str, schema: Type[T]) -> T:
    """Claude: structured output via tool use (fake tool trick)."""
    client = anthropic.Anthropic()
    tool_name = f"extract_{schema.__name__.lower()}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        tools=[{
            "name": tool_name,
            "description": f"Extract {schema.__name__} data from the document",
            "input_schema": schema.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_input = response.content[0].input
    return schema(**tool_input)


def _call_gemini(prompt: str, schema: Type[T]) -> T:
    """Gemini: structured output via native response_schema."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    return response.parsed


PROVIDERS = {
    "claude": _call_claude,
    "gemini": _call_gemini,
}


def call_llm(prompt: str, schema: Type[T]) -> T:
    """Try each provider in order until one succeeds."""
    for provider_name in PROVIDER_ORDER:
        try:
            result = PROVIDERS[provider_name](prompt, schema)
            print(f"  [Using: {provider_name}]")
            return result
        except Exception as e:
            print(f"  [Provider {provider_name} failed: {e}]")

    raise RuntimeError("All LLM providers failed.")


# --- LLM-based document classification ---

class DocType(str, Enum):
    factura = "factura"
    contract = "contract"


class DocumentClassification(BaseModel):
    """Schema for classifying a document."""
    doc_type: DocType = Field(description="The type of document: 'factura' or 'contract'")

# Maps document type to its schema and chunking config.
# chunk_size=None means "don't chunk" (small docs like invoices).
EXTRACTION_REGISTRY = {
    "factura": {
        "schema": Invoice,
        "chunk_size": None,
    },
    "contract": {
        "schema": Contract,
        "chunk_size": 2000,
        "chunk_overlap": 200,
    },
}

# Directory where extracted JSON files will be saved
OUTPUT_DIR = Path(__file__).parent.parent / "extracted_data"


def classify_document(text_preview: str) -> str:
    """Ask the LLM to classify the document type.

    We only send the first 500 chars — that's enough to tell
    an invoice from a contract. This saves tokens and is fast.
    """
    result = call_llm(
        f"Classify this document:\n\n{text_preview[:500]}",
        DocumentClassification,
    )
    return result.doc_type.value


def chunk_text(docs, chunk_size: int, chunk_overlap: int) -> str:
    """Chunk documents and return first 3 chunks combined.

    Why only first 3? Key metadata (parties, dates, contract number)
    is in the header of the document. Using all chunks would add noise
    and waste tokens. (slide 45, lesson 3)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    return "\n\n".join(c.page_content for c in chunks[:3])


def extract(text: str, schema: Type[BaseModel]) -> BaseModel:
    """Send text to LLM and get back a structured Pydantic object.

    Uses structured output — the LLM is forced to return JSON
    matching our schema. No prompt engineering needed for the format,
    Pydantic handles it.
    """
    return call_llm(
        f"Extract the following information from this document:\n\n{text}",
        schema,
    )


def save_json(data: BaseModel, file_path: Path) -> Path:
    """Save extracted Pydantic object as a JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Name the output file: factura_001.txt -> factura_001.json
    output_path = OUTPUT_DIR / f"{file_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)

    return output_path


def process(file_path: str | Path) -> BaseModel:
    """Full extraction pipeline: load -> chunk -> extract -> save JSON.

    This is the main entry point. Give it a file, it returns structured data.
    """
    path = Path(file_path)

    # 1. Load the document
    docs = load_document(path)
    full_text = "\n\n".join(doc.page_content for doc in docs)
    print(f"[Pipeline] Loaded {len(docs)} page(s)")

    # 2. Classify document type using LLM
    doc_type = classify_document(full_text)
    config = EXTRACTION_REGISTRY[doc_type]
    print(f"[Pipeline] Classified as: {doc_type}")

    # 3. Prepare text — chunk or combine directly
    if config["chunk_size"]:
        # Large doc (contract): chunk and use first 3 chunks
        text = chunk_text(docs, config["chunk_size"], config["chunk_overlap"])
        print(f"[Pipeline] Chunked — using first 3 chunks for extraction")
    else:
        # Small doc (invoice): combine all pages directly
        text = "\n\n".join(doc.page_content for doc in docs)
        print(f"[Pipeline] Small doc — using full text")

    # 4. Extract structured data using LLM
    schema = config["schema"]
    data = extract(text, schema)
    print(f"[Pipeline] Extracted: {type(data).__name__}")

    # 5. Save as JSON
    output_path = save_json(data, path)
    print(f"[Pipeline] Saved to: {output_path}")

    # 6. Save to PostgreSQL + embed chunks for RAG
    # RAG chunking is different from extraction chunking:
    # - smaller chunks (500 chars) so each has one clear meaning
    rag_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    rag_chunks = [c.page_content for c in rag_splitter.split_documents(docs)]

    with transaction() as db:
        doc_repo = DocumentRepository(db)
        doc = doc_repo.create(
            filename=path.name,
            content=full_text,
            metadata=data.model_dump(),
        )
        print(f"[Pipeline] Saved to DB: document id={doc.id}")

        rag = RAGService(db)
        rag.add_chunks(doc.id, rag_chunks)
        print(f"[Pipeline] Embedded {len(rag_chunks)} chunks for RAG")

    return data
