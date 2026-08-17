
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter

def split_documents(documents: list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    return chunks

if __name__ == "__main__":

    try:
        from backend.app.ingestion.loader import load_documents
    except ModuleNotFoundError:  # pragma: no cover
        from loader import load_documents

    documents = load_documents()

    chunks = split_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- CHUNK {i + 1} ---")

        print("Metadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content)

source_counts = Counter(
    chunk.metadata.get("source")
    for chunk in chunks
)

for source, count in source_counts.items():
    print(f"{source} → {count} chunks")