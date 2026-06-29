from app.security.sanitizer import DataSanitizer


class StreamingOutputGuard:
    def __init__(self, buffer_size: int = 64, sanitizer: DataSanitizer | None = None) -> None:
        self.buffer_size = buffer_size
        self.sanitizer = sanitizer or DataSanitizer()
        self._buffer = ""
        self._events: list[dict[str, str]] = []

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        if len(self._buffer) <= self.buffer_size:
            return ""

        emit_end = len(self._buffer) - self.buffer_size
        spans = self.sanitizer.detector.detect(self._buffer)
        for span in spans:
            if span.start < emit_end < span.end:
                emit_end = span.start

        if emit_end <= 0:
            return ""

        emit_text = self._buffer[:emit_end]
        self._buffer = self._buffer[emit_end:]
        sanitized = self.sanitizer.sanitize_text(emit_text)
        self._record_mask_events(sanitized.spans)
        return sanitized.sanitized_text

    def flush(self) -> str:
        if not self._buffer:
            return ""
        sanitized = self.sanitizer.sanitize_text(self._buffer)
        self._buffer = ""
        self._record_mask_events(sanitized.spans)
        return sanitized.sanitized_text

    def pop_events(self) -> list[dict[str, str]]:
        events = list(self._events)
        self._events.clear()
        return events

    def _record_mask_events(self, spans: list[object]) -> None:
        for span in spans:
            sensitive_type = getattr(span, "type", None)
            if not sensitive_type:
                continue
            self._events.append(
                {
                    "event_type": "streaming_sensitive_data_masked",
                    "sensitive_type": str(sensitive_type),
                    "source": "sse_output",
                }
            )
