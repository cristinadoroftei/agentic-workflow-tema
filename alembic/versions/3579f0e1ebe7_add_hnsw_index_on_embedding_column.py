"""add hnsw index on embedding column

Revision ID: 3579f0e1ebe7
Revises: dcbab4b1d6b8
Create Date: 2026-06-08 23:35:00.321250

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3579f0e1ebe7'
down_revision: Union[str, Sequence[str], None] = 'dcbab4b1d6b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
