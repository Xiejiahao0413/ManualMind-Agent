from collections import Counter
from pathlib import Path

from app.ingestion.metadata import ChunkMetadataBuilder
from app.ingestion.parser import PDFManualParser, TextManualParser
from app.ingestion.splitter import SectionSplitter
from app.retrieval import HybridRetrieverImpl
from app.schemas.document import DocumentChunk, DocumentIndexResult, ManualDocument


class ManualIndexer:
    def __init__(
        self,
        retriever: HybridRetrieverImpl | None = None,
        text_parser: TextManualParser | None = None,
        pdf_parser: PDFManualParser | None = None,
        splitter: SectionSplitter | None = None,
        metadata_builder: ChunkMetadataBuilder | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetrieverImpl()
        self.text_parser = text_parser or TextManualParser()
        self.pdf_parser = pdf_parser or PDFManualParser()
        self.splitter = splitter or SectionSplitter()
        self.metadata_builder = metadata_builder or ChunkMetadataBuilder(self.splitter)
        self._chunks: list[DocumentChunk] = []
        self._chunks_by_id: dict[str, DocumentChunk] = {}

    def parse_document(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        device_name: str | None = None,
        device_model: str | None = None,
    ) -> ManualDocument:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self.pdf_parser.parse(content, filename, doc_id, device_name, device_model)
        return self.text_parser.parse(content, filename, doc_id, device_name, device_model)

    def index_document(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        device_name: str | None = None,
        device_model: str | None = None,
    ) -> DocumentIndexResult:
        document = self.parse_document(content, filename, doc_id, device_name, device_model)
        sections = self.splitter.split(document)
        chunks = self.metadata_builder.build_chunks(document, sections)
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in self._chunks_by_id]
        if new_chunks:
            self.retriever.add_documents(new_chunks)
            self._chunks.extend(new_chunks)
            self._chunks_by_id.update({chunk.chunk_id: chunk for chunk in new_chunks})

        stats = Counter(chunk.content_type or "general" for chunk in chunks)
        return DocumentIndexResult(
            doc_id=doc_id,
            status="indexed",
            chunks_count=len(chunks),
            content_type_stats=dict(stats),
            sanitized_fields=document.sanitized_fields,
        )

    def list_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def find_chunks(self, chunk_ids: set[str] | None = None) -> list[DocumentChunk]:
        if chunk_ids is None:
            return self.list_chunks()
        return [chunk for chunk_id, chunk in self._chunks_by_id.items() if chunk_id in chunk_ids]

    def has_chunks(self) -> bool:
        return bool(self._chunks)
