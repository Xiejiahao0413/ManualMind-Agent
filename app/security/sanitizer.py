from app.security.schemas import SanitizedText
from app.security.sensitive_detector import SensitiveDataDetector


class DataSanitizer:
    def __init__(self, detector: SensitiveDataDetector | None = None) -> None:
        self.detector = detector or SensitiveDataDetector()

    def sanitize_text(self, text: str) -> SanitizedText:
        spans = self.detector.detect(text)
        sanitized_text = text
        for span in reversed(spans):
            sanitized_text = sanitized_text[: span.start] + span.replacement + sanitized_text[span.end :]
        return SanitizedText(
            original_text=text,
            sanitized_text=sanitized_text,
            spans=spans,
            has_sensitive_data=bool(spans),
        )


def sanitize_text(text: str) -> SanitizedText:
    return DataSanitizer().sanitize_text(text)
