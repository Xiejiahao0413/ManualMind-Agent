from pathlib import Path
from typing import TypedDict

from app.core.dependencies import get_manual_indexer
from app.ingestion import ManualIndexer, UnsupportedPDFTypeError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_MANUALS_DIR = PROJECT_ROOT / "data" / "pdf_manuals"


class IndexedPDFManualSummary(TypedDict):
    indexed_docs_count: int
    total_chunks_count: int
    pages_count: int
    unsupported_files: list[str]
    source_files: list[str]


PDF_DEVICE_CATALOG: dict[str, tuple[str, str]] = {
    "pdf_a100_air_compressor_manual.pdf": ("Air Compressor A100", "A100"),
    "pdf_b200_hydraulic_press_manual.pdf": ("Hydraulic Press B200", "B200"),
}


def index_pdf_manuals(
    pdf_manuals_dir: Path = PDF_MANUALS_DIR,
    indexer: ManualIndexer | None = None,
) -> IndexedPDFManualSummary:
    active_indexer = indexer or get_manual_indexer()
    pdf_paths = sorted(pdf_manuals_dir.glob("*.pdf"))
    indexed_docs_count = 0
    total_chunks_count = 0
    pages_count = 0
    unsupported_files: list[str] = []
    source_files: list[str] = []

    for path in pdf_paths:
        device_name, device_model = infer_pdf_device_metadata(path)
        try:
            active_indexer.index_document(
                content=path.read_bytes(),
                filename=path.name,
                doc_id=path.stem,
                device_name=device_name,
                device_model=device_model,
            )
        except UnsupportedPDFTypeError:
            unsupported_files.append(path.name)
            continue

        indexed_docs_count += 1
        source_files.append(path.name)
        doc_chunks = [chunk for chunk in active_indexer.list_chunks() if chunk.doc_id == path.stem]
        total_chunks_count += len(doc_chunks)
        pages_count += _pages_count(doc_chunks)

    return {
        "indexed_docs_count": indexed_docs_count,
        "total_chunks_count": total_chunks_count,
        "pages_count": pages_count,
        "unsupported_files": unsupported_files,
        "source_files": source_files,
    }


def infer_pdf_device_metadata(path: Path) -> tuple[str, str]:
    catalog_entry = PDF_DEVICE_CATALOG.get(path.name)
    if catalog_entry:
        return catalog_entry
    model = path.stem.split("_", maxsplit=1)[0].upper()
    return model, model


def _pages_count(chunks: list[object]) -> int:
    metadata_pages = [
        int(chunk.metadata["pages_count"])
        for chunk in chunks
        if getattr(chunk, "metadata", {}).get("pages_count") is not None
    ]
    if metadata_pages:
        return max(metadata_pages)
    chunk_pages = [int(chunk.page) for chunk in chunks if getattr(chunk, "page", None)]
    return max(chunk_pages, default=0)
