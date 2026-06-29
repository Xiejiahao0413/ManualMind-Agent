from io import BytesIO
from pathlib import Path
from typing import Any

from app.ingestion.parser import ManualParser
from app.schemas.document import ManualDocument
from app.security.sanitizer import DataSanitizer


PAGE_MARKER_PREFIX = "<!-- page:"


class UnsupportedPDFTypeError(ValueError):
    pass


class PDFManualParser(ManualParser):
    def __init__(self, sanitizer: DataSanitizer | None = None) -> None:
        self.sanitizer = sanitizer or DataSanitizer()

    def parse(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        device_name: str | None = None,
        device_model: str | None = None,
    ) -> ManualDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf_not_installed: install pypdf to parse text-based PDF manuals") from exc

        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:
            raise UnsupportedPDFTypeError(f"unsupported_pdf_type: cannot parse PDF {filename}") from exc

        sanitized_pages: list[str] = []
        sanitized_fields: set[str] = set()
        has_sensitive_data = False
        for page_number, page in enumerate(reader.pages, start=1):
            extracted_text = page.extract_text() or ""
            page_text = extracted_text.strip()
            if not page_text:
                continue
            sanitized = self.sanitizer.sanitize_text(page_text)
            sanitized_fields.update(span.replacement.strip("[]") for span in sanitized.spans)
            has_sensitive_data = has_sensitive_data or sanitized.has_sensitive_data
            sanitized_pages.append(f"<!-- page: {page_number} -->\n{sanitized.sanitized_text}")

        if not sanitized_pages:
            raise UnsupportedPDFTypeError(
                f"unsupported_pdf_type: PDF {filename} has no extractable text; OCR is not supported"
            )

        metadata: dict[str, Any] = {
            "extension": Path(filename).suffix.lower(),
            "has_sensitive_data": has_sensitive_data,
            "pages_count": len(reader.pages),
            "parser": "pypdf",
            "pdf_type": "text_based",
        }
        return ManualDocument(
            doc_id=doc_id,
            filename=filename,
            source_file=filename,
            text="\n\n".join(sanitized_pages),
            device_name=device_name,
            device_model=device_model,
            sanitized_fields=sorted(sanitized_fields),
            metadata=metadata,
        )
