import re
from dataclasses import dataclass

from app.schemas.document import ManualDocument


FAULT_CODE_PATTERN = re.compile(r"\b(?:E|F|P)\d{2,4}\b", re.IGNORECASE)
PAGE_MARKER_PATTERN = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")
HEADING_PATTERN = re.compile(r"^(#{1,6}\s+.+|[一二三四五六七八九十]+[、.].+|\d+(?:\.\d+)*\s+.+)$")
PARAMETER_KEYWORDS = ("温度", "电压", "压力", "阈值", "维护周期", "temperature", "voltage", "pressure", "threshold")
SAFETY_KEYWORDS = ("断电", "高温", "带压", "带电", "禁止", "警告", "安全", "lockout", "power", "pressure")
MAINTENANCE_KEYWORDS = ("维护", "保养", "检修", "更换", "maintenance", "service")


@dataclass(frozen=True)
class Section:
    section_title: str | None
    text: str
    page: int | None = None


class SectionSplitter:
    def split(self, document: ManualDocument) -> list[Section]:
        sections = self._sections_from_headings(document.text)
        if not sections:
            sections = [Section(section_title=None, text=document.text.strip(), page=1)]
        return [section for section in sections if section.text.strip()]

    def detect_content_type(self, text: str, section_title: str | None = None) -> str:
        combined = f"{section_title or ''}\n{text}".lower()
        if FAULT_CODE_PATTERN.search(combined):
            return "fault_code"
        if any(keyword.lower() in combined for keyword in SAFETY_KEYWORDS):
            return "safety_rule"
        if any(keyword.lower() in combined for keyword in PARAMETER_KEYWORDS):
            return "parameter"
        if any(keyword.lower() in combined for keyword in MAINTENANCE_KEYWORDS):
            return "maintenance"
        return "general"

    def detect_fault_code(self, text: str, section_title: str | None = None) -> str | None:
        match = FAULT_CODE_PATTERN.search(f"{section_title or ''}\n{text}")
        return match.group(0).upper() if match else None

    def _sections_from_headings(self, text: str) -> list[Section]:
        sections: list[Section] = []
        current_title: str | None = None
        current_lines: list[str] = []
        page = 1

        for raw_line in text.splitlines():
            line = raw_line.strip()
            page_match = PAGE_MARKER_PATTERN.match(line)
            if page_match:
                self._append_section(sections, current_title, current_lines, page)
                page = int(page_match.group(1))
                current_title = None
                current_lines = []
                continue

            if not line:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue

            if self._is_heading(line):
                self._append_section(sections, current_title, current_lines, page)
                current_title = self._clean_heading(line)
                current_lines = []
                continue

            current_lines.append(raw_line.rstrip())

        self._append_section(sections, current_title, current_lines, page)
        return sections

    def _is_heading(self, line: str) -> bool:
        if HEADING_PATTERN.match(line):
            return len(line) <= 80
        return False

    def _clean_heading(self, line: str) -> str:
        return re.sub(r"^#{1,6}\s*", "", line).strip()

    def _append_section(
        self,
        sections: list[Section],
        title: str | None,
        lines: list[str],
        page: int,
    ) -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(Section(section_title=title, text=text, page=page))
