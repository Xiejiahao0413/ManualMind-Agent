from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import ManualDocument
from app.security.sanitizer import DataSanitizer


class ManualParser(ABC):
    @abstractmethod
    def parse(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        device_name: str | None = None,
        device_model: str | None = None,
    ) -> ManualDocument:
        raise NotImplementedError


class TextManualParser(ManualParser):
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
        text = self._decode(content)
        sanitized = self.sanitizer.sanitize_text(text)
        sanitized_fields = sorted({span.replacement.strip("[]") for span in sanitized.spans})
        return ManualDocument(
            doc_id=doc_id,
            filename=filename,
            source_file=filename,
            text=sanitized.sanitized_text,
            device_name=device_name,
            device_model=device_model,
            sanitized_fields=sanitized_fields,
            metadata={
                "extension": Path(filename).suffix.lower(),
                "has_sensitive_data": sanitized.has_sensitive_data,
            },
        )

    def _decode(self, content: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")


class PDFManualParser(ManualParser):
    def parse(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        device_name: str | None = None,
        device_model: str | None = None,
    ) -> ManualDocument:
        raise NotImplementedError("PDF parsing is reserved for a later implementation pass.")
