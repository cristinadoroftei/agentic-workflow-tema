from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Document


class DocumentRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, filename: str, content: str, metadata: dict) -> Document:
        doc = Document(filename=filename, content=content, doc_metadata=metadata)
        self.db.add(doc)
        self.db.flush()
        return doc

    def get_by_id(self, doc_id: int) -> Document | None:
        return self.db.get(Document, doc_id)

    def get_all(self, limit: int = 100) -> list[Document]:
        return list[Document](self.db.execute(select(Document).limit(limit)).scalars())

    def filter_by_metadata(self, key: str, value: str) -> list[Document]:
        return (
            self.db.query(Document)
            .filter(Document.doc_metadata[key].astext == value)
            .all()
        )
