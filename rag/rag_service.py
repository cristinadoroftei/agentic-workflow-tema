from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from db.repositories import ChunkRepository
from db.models import DocumentChunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGService:

    _model: SentenceTransformer | None = None

    def __init__(self, db: Session) -> None:
        self.chunk_repo = ChunkRepository(db)

    @property
    def model(self) -> SentenceTransformer:
        if RAGService._model is None:
            print(f"[RAG] Loading model {EMBEDDING_MODEL}...")
            RAGService._model = SentenceTransformer(EMBEDDING_MODEL)
        return RAGService._model

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def add_chunks(self, document_id: int, chunks: list[str]) -> list[DocumentChunk]:
        embeddings = self.embed_batch(chunks)
        items = [
            {
                "document_id": document_id,
                "content": content,
                "chunk_index": i,
                "embedding": emb,
            }
            for i, (content, emb) in enumerate(zip(chunks, embeddings))
        ]
        return self.chunk_repo.create_batch(items)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: list[int] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        query_embedding = self.embed(query)
        return self.chunk_repo.similarity_search(query_embedding, top_k=top_k, doc_ids=doc_ids)

    def get_context(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.3,
        doc_ids: list[int] | None = None,
    ) -> str:
        results = self.search(query, top_k=top_k, doc_ids=doc_ids)
        relevant = [(chunk, score) for chunk, score in results if score >= threshold]

        if not relevant:
            return ""

        return "\n\n".join(
            f"[Doc: {chunk.document.filename}, score={score:.2f}]\n{chunk.content}"
            for chunk, score in relevant
        )
