"""
Embedding Model Evaluation with a Separate Test Table.

Uses real pgvector + HNSW indexing (not just numpy in memory)
to evaluate a new embedding model against the current one.

Follows the same patterns as the course code (slide13, slide53, slide57-2).

How to run:
    python embedding_model_evaluation.py
"""

from sqlalchemy import Column, Integer, Text, ForeignKey, Index, select, text
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

from database import Base, engine, transaction
from models import DocumentChunk  # production table (slide53.py)


# ============================================================
# STEP 1: Define a TEST table
# ============================================================
# Same structure as document_chunks, but with a different
# embedding dimension (768 instead of 384) because the new
# model we're testing produces 768-dim vectors.
#
# Production table (slide53.py):
#   document_chunks  →  embedding = Vector(384)   ← old model
#
# Test table:
#   document_chunks_v2_test  →  embedding = Vector(768)  ← new model

class DocumentChunkTest(Base):
    __tablename__ = "document_chunks_v2_test"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(768), nullable=False)  # new model = 768 dim

    __table_args__ = (
        Index("ix_test_chunks_document_id", "document_id"),
    )


# ============================================================
# STEP 2: Create the test table + HNSW index
# ============================================================
# We need an HNSW index so similarity search uses the same
# algorithm as production — not just a sequential scan.

def setup_test_table():
    # Create the table from the SQLAlchemy model above
    Base.metadata.create_all(engine, tables=[DocumentChunkTest.__table__])

    # Create HNSW index (same as production but on the test table)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_test_chunks_embedding_hnsw
            ON document_chunks_v2_test
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        conn.commit()

    print("Test table + HNSW index created.")


# ============================================================
# STEP 3: Copy chunk TEXTS from production, re-embed with new model
# ============================================================
# The chunks (text) stay the same. Only the embeddings change.
#
# Production:  "Penalitati: 0.1% pe zi..."  → [0.12, -0.03, ...]  (384 dim, old model)
# Test table:  "Penalitati: 0.1% pe zi..."  → [0.45, 0.21, ...]  (768 dim, new model)
#                    ↑ same text                  ↑ different vector

def populate_test_table(new_model_name: str = "all-mpnet-base-v2"):
    new_model = SentenceTransformer(new_model_name)

    with transaction() as db:
        # Read all chunk texts from the PRODUCTION table
        production_chunks = db.execute(
            select(
                DocumentChunk.document_id,
                DocumentChunk.content,
                DocumentChunk.chunk_index,
            ).order_by(DocumentChunk.id)
        ).all()

        print(f"Read {len(production_chunks)} chunks from production table.")

        # Embed the same texts with the NEW model
        texts = [chunk.content for chunk in production_chunks]
        new_embeddings = new_model.encode(texts, show_progress_bar=True)

        # Insert into the test table (same text, new embeddings)
        test_rows = [
            DocumentChunkTest(
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                embedding=new_embeddings[i].tolist(),
            )
            for i, chunk in enumerate(production_chunks)
        ]
        db.add_all(test_rows)

    print(f"Inserted {len(test_rows)} chunks into test table with {new_model_name}.")


# ============================================================
# STEP 4: Repositories for both tables
# ============================================================
# Production repo: searches document_chunks with old embeddings
# Test repo:       searches document_chunks_v2_test with new embeddings
#
# Both use the same similarity_search pattern from slide57-2.py

class ProductionChunkRepository:
    """Searches the production table (384-dim, old model)."""

    def __init__(self, db):
        self.db = db

    def similarity_search(self, query_embedding, top_k=5):
        similarity = (
            1 - DocumentChunk.embedding.cosine_distance(query_embedding)
        ).label("score")

        stmt = (
            select(DocumentChunk, similarity)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()
        return [(chunk, float(score)) for chunk, score in rows]


class TestChunkRepository:
    """Searches the test table (768-dim, new model)."""

    def __init__(self, db):
        self.db = db

    def similarity_search(self, query_embedding, top_k=5):
        similarity = (
            1 - DocumentChunkTest.embedding.cosine_distance(query_embedding)
        ).label("score")

        stmt = (
            select(DocumentChunkTest, similarity)
            .order_by(DocumentChunkTest.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()
        return [(chunk, float(score)) for chunk, score in rows]


# ============================================================
# STEP 5: The validation set (human-curated ground truth)
# ============================================================
# A human looked at the chunks and decided:
#   "For this question, these specific chunks contain the answer."
#
# This is built ONCE, then reused for every model comparison.

VALIDATION_SET = [
    {
        "question": "Care sunt penalitatile de intarziere?",
        "document_id": 1,
        "golden_chunk_indexes": [4, 5],
    },
    {
        "question": "Cine sunt partile contractante?",
        "document_id": 1,
        "golden_chunk_indexes": [0, 1],
    },
    {
        "question": "Ce spune contractul despre forta majora?",
        "document_id": 1,
        "golden_chunk_indexes": [6],
    },
    # In practice: 50-100 questions for a reliable evaluation
]


# ============================================================
# STEP 6: Run the evaluation
# ============================================================
# For each validation question:
#   - Embed the question with the OLD model → search PRODUCTION table
#   - Embed the question with the NEW model → search TEST table
#   - Check how many golden chunks appear in the top_k results
#   - Calculate recall and precision
#
# IMPORTANT: we embed the question with the MATCHING model.
#   Old model embeds → search old table  (same vector space)
#   New model embeds → search new table  (same vector space)
# You can't mix them — different models produce incompatible vectors.

def evaluate(top_k: int = 5):
    old_model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dim, current
    new_model = SentenceTransformer("all-mpnet-base-v2")   # 768 dim, candidate

    old_recalls = []
    new_recalls = []
    old_precisions = []
    new_precisions = []

    with transaction() as db:
        old_repo = ProductionChunkRepository(db)
        new_repo = TestChunkRepository(db)

        for item in VALIDATION_SET:
            question = item["question"]
            doc_id = item["document_id"]
            golden = set(item["golden_chunk_indexes"])

            # --- Old model: embed question → search production table ---
            q_old = old_model.encode(question).tolist()
            old_results = old_repo.similarity_search(q_old, top_k=top_k)
            old_found = {
                chunk.chunk_index for chunk, score in old_results
                if chunk.document_id == doc_id
            }

            # --- New model: embed question → search test table ---
            q_new = new_model.encode(question).tolist()
            new_results = new_repo.similarity_search(q_new, top_k=top_k)
            new_found = {
                chunk.chunk_index for chunk, score in new_results
                if chunk.document_id == doc_id
            }

            # --- Recall: of all golden chunks, how many did we find? ---
            old_recall = len(old_found & golden) / len(golden)
            new_recall = len(new_found & golden) / len(golden)
            old_recalls.append(old_recall)
            new_recalls.append(new_recall)

            # --- Precision: of results returned, how many are golden? ---
            old_precision = len(old_found & golden) / top_k if top_k > 0 else 0
            new_precision = len(new_found & golden) / top_k if top_k > 0 else 0
            old_precisions.append(old_precision)
            new_precisions.append(new_precision)

            print(f"\nQ: {question}")
            print(f"  Golden:    {golden}")
            print(f"  Old found: {old_found}  recall={old_recall:.2f}  precision={old_precision:.2f}")
            print(f"  New found: {new_found}  recall={new_recall:.2f}  precision={new_precision:.2f}")

    # --- Average across ALL validation questions ---
    avg_old_recall = sum(old_recalls) / len(old_recalls)
    avg_new_recall = sum(new_recalls) / len(new_recalls)
    avg_old_precision = sum(old_precisions) / len(old_precisions)
    avg_new_precision = sum(new_precisions) / len(new_precisions)

    print(f"\n{'=' * 60}")
    print(f"RESULTS — averaged over {len(VALIDATION_SET)} questions (top_k={top_k})")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'Old (MiniLM-L6)':>16} {'New (mpnet-base)':>16}")
    print(f"  {'-' * 52}")
    print(f"  {'Recall@' + str(top_k):<20} {avg_old_recall:>15.2%} {avg_new_recall:>15.2%}")
    print(f"  {'Precision@' + str(top_k):<20} {avg_old_precision:>15.2%} {avg_new_precision:>15.2%}")
    print()

    diff = avg_new_recall - avg_old_recall
    if diff > 0.05:
        print(f"  >> New model is better by {diff:.2%}. Consider migrating.")
    elif diff > 0:
        print(f"  >> New model is slightly better by {diff:.2%}. Probably not worth it.")
    else:
        print(f"  >> Old model is equal or better. Keep it.")


# ============================================================
# STEP 7: Cleanup — drop the test table when done
# ============================================================

def cleanup_test_table():
    DocumentChunkTest.__table__.drop(engine)
    print("Test table dropped.")


# ============================================================
# Main: run the full evaluation pipeline
# ============================================================

if __name__ == "__main__":
    # 1. Create the test table with HNSW index
    setup_test_table()

    # 2. Copy chunks from production, embed with new model
    populate_test_table("all-mpnet-base-v2")

    # 3. Evaluate at different top_k values
    evaluate(top_k=3)
    evaluate(top_k=5)

    # 4. Drop the test table (uncomment when done)
    # cleanup_test_table()
