from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from db.models import DocumentChunk


class ChunkRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, document_id: int, content: str, chunk_index: int, embedding: list[float]) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            embedding=embedding,
        )
        self.db.add(chunk)
        self.db.flush()
        return chunk

    def create_batch(self, items: list[dict]) -> list[DocumentChunk]:
        chunks = [DocumentChunk(**item) for item in items]
        self.db.add_all(chunks)
        self.db.flush()
        return chunks

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        similarity = (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label("score")

        stmt = (
            select(DocumentChunk, similarity)
            .options(joinedload(DocumentChunk.document))
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        rows = self.db.execute(stmt).all()
        return [(chunk, float(score)) for chunk, score in rows]
