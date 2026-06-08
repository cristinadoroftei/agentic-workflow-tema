from pydantic import BaseModel, Field


class CalculatorParams(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate (e.g. '2 + 3 * 4')",
        min_length=1,
    )


class GetDatetimeParams(BaseModel):
    timezone: str = Field(
        default="UTC",
        description="Desired timezone (e.g. 'Europe/Bucharest', 'UTC')",
    )


class WebSearchParams(BaseModel):
    query: str = Field(
        description="Search query terms",
        min_length=2,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20,
    )


class SearchDocumentsParams(BaseModel):
    query: str = Field(
        description="Natural language question to search for in stored documents (invoices, contracts)",
        min_length=2,
    )
    top_k: int = Field(
        default=3,
        description="Number of most relevant chunks to return",
        ge=1,
        le=10,
    )
    doc_type: str | None = Field(
        default=None,
        description="Filter by document type before searching: 'factura' or 'contract'. Leave empty to search all documents.",
    )
