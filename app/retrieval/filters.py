from typing import Any

from app.schemas.retrieval import DocumentChunk, RetrievalResult

SUPPORTED_METADATA_FILTERS = {
    "device_name",
    "device_model",
    "content_type",
    "fault_code",
    "doc_id",
}


def matches_metadata_filter(
    item: DocumentChunk | RetrievalResult,
    metadata_filter: dict[str, Any] | None = None,
) -> bool:
    if not metadata_filter:
        return True

    for key, expected in metadata_filter.items():
        if key not in SUPPORTED_METADATA_FILTERS:
            continue
        actual = getattr(item, key)
        if isinstance(expected, list | tuple | set):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
