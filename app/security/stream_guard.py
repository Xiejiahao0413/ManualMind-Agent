from app.security.sanitizer import DataSanitizer


class StreamingOutputGuard:
    def __init__(self, buffer_size: int = 64, sanitizer: DataSanitizer | None = None) -> None:
        self.buffer_size = buffer_size
        self.sanitizer = sanitizer or DataSanitizer()
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        combined = self._buffer + chunk
        if len(combined) <= self.buffer_size:
            self._buffer = combined
            return ""

        emit_text = combined[: -self.buffer_size]
        self._buffer = combined[-self.buffer_size :]
        return self.sanitizer.sanitize_text(emit_text).sanitized_text

    def flush(self) -> str:
        if not self._buffer:
            return ""
        sanitized = self.sanitizer.sanitize_text(self._buffer).sanitized_text
        self._buffer = ""
        return sanitized
