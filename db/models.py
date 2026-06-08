from typing import Any


from datetime import datetime


from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint, Index, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column[int](Integer, primary_key=True, autoincrement=True)
    filename = Column[str](String(255), nullable=False, index=True)
    content = Column[str](Text, nullable=False)
    # "metadata" is reserved in SQLAlchemy, so the attribute is doc_metadata
    # but the actual PostgreSQL column is called "metadata"
    doc_metadata = Column[Any]("metadata", JSONB, nullable=False, default=dict)
    created_at = Column[datetime](DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename='{self.filename}'>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column[int](Integer, primary_key=True, autoincrement=True)
    document_id = Column[int](
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content = Column[str](Text, nullable=False)
    chunk_index = Column[int](Integer, nullable=False)
    embedding = Column[Any](Vector(384), nullable=False)  # all-MiniLM-L6-v2 = 384 dimensions

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_idx"),
        Index("ix_chunks_document_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} chunk={self.chunk_index}>"
