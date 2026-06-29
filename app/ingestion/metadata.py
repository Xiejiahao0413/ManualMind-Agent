import hashlib
import re

from app.ingestion.splitter import Section, SectionSplitter
from app.schemas.document import DocumentChunk, ManualDocument, normalize_metadata


PAGE_NUMBER_TITLE_PATTERN = re.compile(
    r"^(?:\d+\s*/\s*\d+|\d+|第\s*\d+\s*页|page\s*\d+)(?:\s*/\s*\d+)?$",
    re.IGNORECASE,
)


class ChunkMetadataBuilder:
    def __init__(self, splitter: SectionSplitter | None = None) -> None:
        self.splitter = splitter or SectionSplitter()

    def build_chunks(self, document: ManualDocument, sections: list[Section]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section in sections:
            section_title = self.clean_section_title(section.section_title)
            content_type = self.splitter.detect_content_type(section.text, section_title)
            fault_code = self.splitter.detect_fault_code(section.text, section_title)
            chunk_id = self.build_chunk_id(
                doc_id=document.doc_id,
                section_title=section_title,
                content_type=content_type,
                fault_code=fault_code,
                text=section.text,
            )
            metadata = normalize_metadata(
                {
                    **document.metadata,
                    "sanitized_fields": ",".join(document.sanitized_fields),
                }
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    device_name=document.device_name,
                    device_model=document.device_model,
                    section_title=section_title,
                    page=section.page,
                    content_type=content_type,
                    fault_code=fault_code,
                    source_file=document.source_file or document.filename,
                    text=section.text,
                    metadata=metadata,
                )
            )
        return chunks

    def clean_section_title(self, section_title: str | None) -> str | None:
        if not section_title:
            return None
        cleaned = section_title.strip()
        if PAGE_NUMBER_TITLE_PATTERN.match(cleaned):
            return None
        return cleaned

    def build_chunk_id(
        self,
        doc_id: str,
        section_title: str | None,
        content_type: str,
        fault_code: str | None,
        text: str,
    ) -> str:
        stable_value = "|".join(
            [
                doc_id,
                section_title or "",
                content_type,
                fault_code or "",
                text.strip(),
            ]
        )
        digest = hashlib.sha1(stable_value.encode("utf-8")).hexdigest()[:16]
        return f"{doc_id}-{digest}"
