import json
import logging
from pathlib import Path
from typing import Callable
from .metadata import create_metadata
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
LOGGER = logging.getLogger(__name__)

LoaderFn = Callable[[Path], list[Document]]


def _load_text_file(file_path: Path) -> list[Document]:
    loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    return loader.load()


def _load_csv_file(file_path: Path) -> list[Document]:
    loader = CSVLoader(str(file_path), encoding="utf-8")
    return loader.load()


def _load_pdf_file(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def _json_to_documents(file_path: Path) -> list[Document]:
    with file_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if isinstance(data, list):
        return [
            Document(
                page_content=json.dumps(item, ensure_ascii=False, indent=2),
                metadata={"source": str(file_path), "record_index": idx},
            )
            for idx, item in enumerate(data)
        ]

    return [
        Document(
            page_content=json.dumps(data, ensure_ascii=False, indent=2),
            metadata={"source": str(file_path)},
        )
    ]


LOADER_REGISTRY: dict[str, LoaderFn] = {
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".csv": _load_csv_file,
    ".pdf": _load_pdf_file,
    ".json": _json_to_documents,
}


def register_loader(extension: str, loader_fn: LoaderFn) -> None:
    """Register or replace a loader for an extension (example: '.docx')."""
    normalized = extension.lower().strip()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    LOADER_REGISTRY[normalized] = loader_fn


def load_documents(
    base_dir: Path = KNOWLEDGE_BASE_DIR,
    include_extensions: set[str] | None = None,
    raise_on_error: bool = False,
) -> list[Document]:
    """Load supported documents from a directory tree.

    Args:
        base_dir: Root folder to scan recursively.
        include_extensions: Optional whitelist of extensions.
        raise_on_error: Raise ingestion errors if True, else skip invalid files.
    """
    documents: list[Document] = []
    selected_extensions = {ext.lower() for ext in include_extensions} if include_extensions else None

    for file_path in sorted(base_dir.rglob("*")):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in LOADER_REGISTRY:
            continue
        if selected_extensions and extension not in selected_extensions:
            continue

        try:
            loaded_docs = LOADER_REGISTRY[extension](file_path)

            for doc in loaded_docs:
                doc.metadata.update(create_metadata(file_path))

            documents.extend(loaded_docs)
        except Exception as exc:
            message = f"Failed to load {file_path}: {exc}"
            if raise_on_error:
                raise RuntimeError(message) from exc
            LOGGER.warning(message)

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    for doc in docs[:10]:
        print("\nSOURCE:")
        print(doc.metadata["source"])

        print("\nMETADATA:")
        print(doc.metadata)

        print("\nCONTENT:")
        print(doc.page_content[:100])


