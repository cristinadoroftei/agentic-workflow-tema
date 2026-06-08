from datetime import datetime

import httpx
import zoneinfo

from tools.registry import register_tool
from tools.params_models import CalculatorParams, GetDatetimeParams, WebSearchParams, SearchDocumentsParams

from db.database import transaction
from db.repositories import DocumentRepository
from rag.rag_service import RAGService


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluates a simple mathematical expression and returns the result."""
    # Only allow safe characters: digits, operators, parentheses, spaces, dots
    allowed = set[str]("0123456789+-*/.() ")
    if not all(c in allowed for c in params.expression):
        return f"Error: expression contains invalid characters."

    result = eval(params.expression)
    return str(result)


@register_tool
def get_datetime(params: GetDatetimeParams) -> str:
    """Returns the current date and time in the specified timezone."""
    tz = zoneinfo.ZoneInfo(params.timezone)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@register_tool
def web_search(params: WebSearchParams) -> str:
    """Searches the web for information and returns a list of results."""
    response = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": params.query, "format": "json", "no_html": 1},
        headers={"User-Agent": "QAAgent/1.0"},
        timeout=10.0,
    )
    data = response.json()

    results = []
    # AbstractText is DuckDuckGo's instant answer
    if data.get("AbstractText"):
        results.append(f"Summary: {data['AbstractText']}")

    # Collect topics — some are direct, some are nested in sub-groups
    all_topics = []
    for topic in data.get("RelatedTopics", []):
        if "Text" in topic:
            all_topics.append(topic["Text"])
        elif "Topics" in topic:
            for sub in topic["Topics"]:
                if "Text" in sub:
                    all_topics.append(sub["Text"])

    results.extend(all_topics[:params.max_results])

    if not results:
        return f"No results found for '{params.query}'. Try a broader or simpler query."

    return "\n".join(results)


@register_tool
def search_documents(params: SearchDocumentsParams) -> str:
    """Searches stored documents (invoices, contracts) for relevant information using semantic similarity. Use this tool when the user asks about document contents, suppliers, clients, amounts, dates, or contract clauses. Set doc_type to 'factura' or 'contract' to narrow the search when the user's question is clearly about one type."""

    with transaction() as db:
        doc_ids = None
        if params.doc_type:
            doc_repo = DocumentRepository(db)
            docs = doc_repo.filter_by_metadata("doc_type", params.doc_type)
            doc_ids = [d.id for d in docs]
            if not doc_ids:
                return f"No documents found with type '{params.doc_type}'."

        rag = RAGService(db)
        context = rag.get_context(params.query, top_k=params.top_k, doc_ids=doc_ids)

    if not context:
        return "No relevant documents found for this query."

    return context
