from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document

# --- Registry: maps file extension to the right loader class ---
# Each loader knows how to parse its format, but they ALL return List[Document].
# This is the power of LangChain's abstraction — uniform interface regardless of format.
LOADER_REGISTRY = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".csv":  CSVLoader,
}


def load_document(file_path: str | Path) -> list[Document]:
    """Load a document using the appropriate loader based on file extension.

    Args:
        file_path: Path to the document file (PDF, DOCX, or TXT)

    Returns:
        List of Document objects (each has .page_content and .metadata)

    Raises:
        ValueError: If the file extension is not supported
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Look up the loader class from the registry
    extension = path.suffix.lower()
    loader_class = LOADER_REGISTRY.get(extension)

    if loader_class is None:
        supported = ", ".join(LOADER_REGISTRY.keys())
        raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")

    # Instantiate the loader and load the document
    loader = loader_class(str(path))
    return loader.load()
