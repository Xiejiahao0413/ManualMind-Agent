from pathlib import Path
from typing import TypedDict

from app.ingestion import ManualIndexer
from app.schemas.retrieval import DocumentChunk
from scripts.manuals import MANUALS_DIR, index_manuals
from scripts.pdf_manuals import PDF_MANUALS_DIR, index_pdf_manuals


class VectorManualCorpus(TypedDict):
    chunks: list[DocumentChunk]
    source_files: list[str]
    device_models: list[str]


def load_synthetic_manual_chunks(
    manuals_dir: Path = MANUALS_DIR,
    pdf_manuals_dir: Path = PDF_MANUALS_DIR,
) -> VectorManualCorpus:
    indexer = ManualIndexer()
    index_manuals(manuals_dir, indexer=indexer)
    index_pdf_manuals(pdf_manuals_dir, indexer=indexer)
    chunks = indexer.list_chunks()
    return {
        "chunks": chunks,
        "source_files": sorted({chunk.source_file for chunk in chunks if chunk.source_file}),
        "device_models": sorted({chunk.device_model for chunk in chunks if chunk.device_model}),
    }
