from pathlib import Path

DOCUMENT_TYPE_MAP = {
    "company": "company_information",
    "capabilities": "capability",
    "case_studies": "case_study",
    "pricing": "pricing",
}


def create_metadata(file_path: Path) -> dict[str, str]:

    category = file_path.parent.name

    metadata = {
        "source": str(file_path),
        "file_name": file_path.name,
        "category": file_path.parent.name,
        "document_type": DOCUMENT_TYPE_MAP.get(category, "unknown"),
    }
    return metadata