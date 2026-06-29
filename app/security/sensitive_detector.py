import re
from dataclasses import dataclass
from re import Pattern

from app.security.schemas import SensitiveSpan

REPLACEMENTS: dict[str, str] = {
    "phone_number": "[PHONE]",
    "email": "[EMAIL]",
    "ip_address": "[IP_ADDRESS]",
    "device_id": "[DEVICE_ID]",
    "work_order_id": "[WORK_ORDER_ID]",
    "internal_url": "[INTERNAL_URL]",
    "location_keyword": "[LOCATION]",
}


@dataclass(frozen=True)
class SensitivePattern:
    type: str
    pattern: Pattern[str]


LOCATION_KEYWORDS = (
    "Beijing",
    "Shanghai",
    "Shenzhen",
    "Guangzhou",
    "\u5317\u4eac",
    "\u4e0a\u6d77",
    "\u6df1\u5733",
    "\u5e7f\u5dde",
    "\u673a\u623f",
    "\u5de5\u5382",
)


class SensitiveDataDetector:
    def __init__(self) -> None:
        self.patterns: tuple[SensitivePattern, ...] = (
            SensitivePattern(
                "internal_url",
                re.compile(
                    r"https?://(?:localhost|127\.0\.0\.1|(?:10|192\.168|172\.(?:1[6-9]|2\d|3[0-1]))\.[^\s/]+|[A-Za-z0-9.-]*(?:internal|intranet|corp|local)[A-Za-z0-9.-]*)(?:[^\s]*)",
                    re.IGNORECASE,
                ),
            ),
            SensitivePattern(
                "email",
                re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
            ),
            SensitivePattern(
                "phone_number",
                re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)"),
            ),
            SensitivePattern(
                "ip_address",
                re.compile(
                    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
                ),
            ),
            SensitivePattern(
                "work_order_id",
                re.compile(
                    r"\b(?:WO|WORKORDER)(?:[-_]?[A-Z0-9]{3,})+\b|(?:\u5de5\u5355(?:\u53f7)?[:\uff1a]?\s*)[A-Z0-9-]{4,}",
                    re.IGNORECASE,
                ),
            ),
            SensitivePattern(
                "device_id",
                re.compile(
                    r"\b(?:DEVICE|DEV|EQP|SN)(?:[-_][A-Z0-9]{3,}(?:[-_][A-Z0-9]{2,})*|[0-9][A-Z0-9]{2,})\b",
                    re.IGNORECASE,
                ),
            ),
        )

    def detect(self, text: str) -> list[SensitiveSpan]:
        spans: list[SensitiveSpan] = []
        for sensitive_pattern in self.patterns:
            for match in sensitive_pattern.pattern.finditer(text):
                spans.append(
                    SensitiveSpan(
                        type=sensitive_pattern.type,
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        replacement=REPLACEMENTS[sensitive_pattern.type],
                    )
                )

        spans.extend(self._detect_location_keywords(text))
        return self._dedupe_overlaps(spans)

    def _detect_location_keywords(self, text: str) -> list[SensitiveSpan]:
        spans: list[SensitiveSpan] = []
        lower_text = text.lower()
        for keyword in LOCATION_KEYWORDS:
            search_from = 0
            lower_keyword = keyword.lower()
            while True:
                start = lower_text.find(lower_keyword, search_from)
                if start == -1:
                    break
                end = start + len(keyword)
                spans.append(
                    SensitiveSpan(
                        type="location_keyword",
                        value=text[start:end],
                        start=start,
                        end=end,
                        replacement=REPLACEMENTS["location_keyword"],
                    )
                )
                search_from = end
        return spans

    def _dedupe_overlaps(self, spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
        sorted_spans = sorted(spans, key=lambda span: (span.start, -(span.end - span.start)))
        accepted: list[SensitiveSpan] = []
        for span in sorted_spans:
            if any(span.start < existing.end and span.end > existing.start for existing in accepted):
                continue
            accepted.append(span)
        return sorted(accepted, key=lambda span: span.start)
