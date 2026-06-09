import re
from typing import Any

from app.schemas.diagnosis import DiagnosisState


ENGLISH_TEMPLATE_SENTENCES = (
    "Review retrieved manual evidence.",
    "Check the fault code or parameter range against the cited source.",
    "Record observations before corrective action.",
    "No high-risk operation detected by current tool results.",
    "Manual evidence suggests checking device-specific operating conditions.",
)


def build_diagnosis_report(state: DiagnosisState) -> str:
    fault_text = _fault_description(state)
    causes = _extract_section_items(fault_text, "可能原因", ("排查步骤", "安全提醒", "技术参数", "引用来源"))
    steps = _extract_section_items(fault_text, "排查步骤", ("安全提醒", "可能原因", "技术参数", "引用来源"))
    safety_notices = _extract_safety_notices(state, fault_text)
    source_refs = _dedupe([str(source) for source in state.source_refs if source])

    if not causes:
        causes = _fallback_causes(state)
    if not steps:
        steps = _fallback_steps(state)
    if not safety_notices:
        safety_notices = [
            "当前未识别到高风险维修操作，建议仍按照设备手册要求进行断电和安全检查。"
        ]

    return "\n\n".join(
        [
            "故障识别：\n" + _clean_text(_fault_summary(state, fault_text)),
            "可能原因：\n" + _format_bullets(causes),
            "排查步骤：\n" + _format_numbered(steps),
            "安全提醒：\n" + _format_bullets(safety_notices),
            "引用来源：\n" + _format_bullets(source_refs or ["暂无引用来源"]),
        ]
    )


def _fault_description(state: DiagnosisState) -> str:
    if state.fault_info and state.fault_info.get("description"):
        return str(state.fault_info["description"])
    for chunk in state.retrieved_chunks:
        if chunk.get("content_type") == "fault_code" and chunk.get("text"):
            return str(chunk["text"])
    if state.fault_code:
        return f"识别到故障码 {state.fault_code}，请结合设备手册和检索证据进行排查。"
    return "未识别到明确故障码，需结合设备症状和手册证据继续排查。"


def _fault_summary(state: DiagnosisState, text: str) -> str:
    cleaned = _strip_sections(text, ("可能原因", "排查步骤", "安全提醒", "引用来源"))
    cleaned = _first_nonempty_line(cleaned)
    if cleaned:
        return cleaned
    if state.fault_code:
        return f"识别到故障码 {state.fault_code}。"
    return "当前未识别到明确故障码。"


def _extract_safety_notices(state: DiagnosisState, fault_text: str) -> list[str]:
    notices: list[str] = []
    notices.extend(_extract_section_items(fault_text, "安全提醒", ("可能原因", "排查步骤", "引用来源")))
    notices.extend(str(rule) for rule in state.safety_rules if rule)
    for chunk in state.retrieved_chunks:
        if chunk.get("content_type") == "safety_rule" and chunk.get("text"):
            notices.append(str(chunk["text"]))
    return _dedupe(_clean_text(item) for item in notices if item)


def _fallback_causes(state: DiagnosisState) -> list[str]:
    if state.parameter_info:
        return ["参数信息存在异常或需要结合设备手册确认。"]
    if state.fault_code:
        return ["故障原因需结合故障码说明、设备状态和现场检查结果确认。"]
    return ["设备症状需要结合手册证据和现场状态进一步确认。"]


def _fallback_steps(state: DiagnosisState) -> list[str]:
    if state.fault_code:
        return [
            f"核对设备显示的故障码是否为 {state.fault_code}。",
            "查看引用来源中的故障说明和排查要求。",
            "记录现场现象、处理动作和复测结果。",
        ]
    return [
        "查看引用来源中的相关手册内容。",
        "核对设备状态、参数和报警信息。",
        "记录现场现象、处理动作和复测结果。",
    ]


def _extract_section_items(text: str, title: str, stop_titles: tuple[str, ...]) -> list[str]:
    pattern = re.compile(
        rf"{re.escape(title)}\s*[:：]\s*(?P<body>.*?)(?=(?:{'|'.join(re.escape(item) for item in stop_titles)})\s*[:：]|$)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    body = match.group("body").strip()
    return _dedupe(_split_items(body))


def _split_items(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<!\n)(\d+[.、]\s*)", r"\n\1", normalized)
    normalized = re.sub(r"(?<!\n)([-*]\s*)", r"\n\1", normalized)
    items: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+[.、]\s*", "", line)
        line = _clean_text(line)
        if line:
            items.append(line)
    return items


def _strip_sections(text: str, titles: tuple[str, ...]) -> str:
    if not text:
        return ""
    pattern = re.compile(rf"\s*(?:{'|'.join(re.escape(title) for title in titles)})\s*[:：].*$", re.DOTALL)
    return pattern.sub("", text).strip()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = _clean_text(line)
        if cleaned:
            return cleaned
    return ""


def _clean_text(text: Any) -> str:
    cleaned = str(text).strip()
    for sentence in ENGLISH_TEMPLATE_SENTENCES:
        cleaned = cleaned.replace(sentence, "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" 。")


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in _dedupe(items))


def _format_numbered(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(_dedupe(items), start=1))


def _dedupe(items) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
