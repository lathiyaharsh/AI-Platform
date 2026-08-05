from app.rag.types import ChunkStrategy, VectorStoreBackend

# Default pipeline / retrieval knobs (overridden by Settings).
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.55
DEFAULT_CHUNK_STRATEGY = ChunkStrategy.SENTENCE
DEFAULT_VECTOR_STORE = VectorStoreBackend.MEMORY

# Text cleaning
MAX_NULL_REPLACEMENTS = True

# Extensions → DocumentType mapping (shared by loaders registry).
EXTENSION_MAP: dict[str, str] = {
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".pdf": "pdf",
    # Reserved for future loaders:
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
}

SUPPORTED_UPLOAD_EXTENSIONS = frozenset({".txt", ".text", ".md", ".markdown", ".pdf"})
