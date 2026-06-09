"""Verify database contents after running the pipeline.

Usage: python3 verify_db.py
"""

from db.database import transaction
from db.repositories import DocumentRepository, ChunkRepository
from rag.rag_service import RAGService


def main() -> None:
    with transaction() as db:
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)

        # 1. Documents table
        docs = doc_repo.get_all()
        print(f"=== documents table: {len(docs)} rows ===\n")
        for doc in docs:
            meta = doc.doc_metadata or {}
            doc_type = meta.get("doc_type", "?")
            print(f"  id={doc.id}  type={doc_type}  filename={doc.filename}")
            # Show a few metadata fields depending on type
            if doc_type == "factura":
                print(f"    furnizor={meta.get('furnizor')}  total={meta.get('total')} {meta.get('moneda')}")
            elif doc_type == "contract":
                print(f"    prestator={meta.get('prestator')}  valoare={meta.get('valoare_totala')} {meta.get('moneda')}")

        # 2. Chunks table
        total_chunks = sum(len(doc.chunks) for doc in docs)
        print(f"\n=== document_chunks table: {total_chunks} rows ===\n")
        for doc in docs:
            print(f"  document_id={doc.id} ({doc.filename}): {len(doc.chunks)} chunks")

        # 3. Hybrid search test
        print("\n=== hybrid search test ===\n")
        rag = RAGService(db)

        # Search only in contracts
        contracts = doc_repo.filter_by_metadata("doc_type", "contract")
        contract_ids = [d.id for d in contracts]
        results = rag.search("penalitati", top_k=2, doc_ids=contract_ids)
        print(f"  query='penalitati', filter=contract only ({len(contracts)} docs)")
        for chunk, score in results:
            print(f"    score={score:.2f}  doc={chunk.document.filename}  chunk={chunk.chunk_index}")

        # Search all documents
        results = rag.search("cine este furnizorul", top_k=2)
        print(f"\n  query='cine este furnizorul', filter=none (all docs)")
        for chunk, score in results:
            print(f"    score={score:.2f}  doc={chunk.document.filename}  chunk={chunk.chunk_index}")


if __name__ == "__main__":
    main()
