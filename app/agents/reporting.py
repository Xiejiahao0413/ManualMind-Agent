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


def build_manual_qa_answer(state: DiagnosisState) -> str:
    chunks = [
        chunk
        for chunk in state.retrieved_chunks
        if _clean_text(chunk.get("text") or "")
    ]
    if not chunks:
        return "未在上传手册中找到依据。请确认上传的手册是否已完成索引，或补充更具体的章节、功能名称后重试。"

    points: list[str] = []
    notes: list[str] = []
    for chunk in chunks[:3]:
        text = _clean_text(chunk.get("text") or "")
        extracted = _split_items(text)
        if not extracted and text:
            extracted = _split_sentences(text)
        for item in extracted:
            if _looks_like_operation_step(item):
                points.append(item)
            else:
                notes.append(item)

    if not points:
        points = notes[:5]
        notes = notes[5:]

    source_refs = _manual_qa_source_refs(chunks, state.source_refs)

    sections = [
        "根据上传手册，相关操作说明如下：\n\n" + _format_numbered(points[:8]),
    ]
    if notes:
        sections.append("补充说明：\n" + _format_bullets(notes[:5]))
    sections.append("引用来源：\n" + _format_bullets(source_refs or ["暂无引用来源"]))
    return "\n\n".join(sections)


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
    fault_chunks = [
        chunk
        for chunk in state.retrieved_chunks
        if chunk.get("content_type") == "fault_code"
        and chunk.get("text")
        and chunk.get("section_title") != "故障码表"
    ]
    if state.fault_code:
        for chunk in fault_chunks:
            if str(chunk.get("fault_code") or "").upper() == state.fault_code:
                return str(chunk["text"])
    else:
        matched_chunks = sorted(
            fault_chunks,
            key=lambda chunk: _query_overlap_score(state, chunk),
            reverse=True,
        )
        if matched_chunks and _query_overlap_score(state, matched_chunks[0]) > 0:
            return str(matched_chunks[0]["text"])
    if state.fault_code:
        return f"识别到故障码 {state.fault_code}，请结合设备手册和检索证据进行排查。"
    return "未识别到明确故障码，需结合设备症状和手册证据继续排查。"


def _query_overlap_score(state: DiagnosisState, chunk: dict) -> int:
    text = f"{chunk.get('section_title') or ''}\n{chunk.get('text') or ''}".lower()
    return sum(1 for term in _query_terms(state) if term in text)


def _query_terms(state: DiagnosisState) -> list[str]:
    query = (state.raw_query or state.sanitized_query or state.user_query or "").lower()
    terms = re.findall(r"[a-z]+\d+|\d+[a-z]+|[a-z0-9_-]+|[\u4e00-\u9fff]{2,}", query)
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", term):
            expanded.extend(term[index : index + 2] for index in range(len(term) - 1))
    stop_terms = {"怎么", "处理", "排查", "原因", "如何", "什么", "设备"}
    return [term for term in expanded if term not in stop_terms]


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


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[。！？.!?])\s*|[；;]\s*", normalized)
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _looks_like_operation_step(text: str) -> bool:
    lower_text = text.lower()
    operation_terms = (
        "点击",
        "选择",
        "输入",
        "设置",
        "创建",
        "新建",
        "保存",
        "确认",
        "打开",
        "进入",
        "配置",
        "执行",
        "按",
        "click",
        "select",
        "enter",
        "set",
        "create",
        "save",
        "open",
        "configure",
    )
    return any(term in lower_text for term in operation_terms)


def _manual_qa_source_refs(chunks: list[dict], fallback_refs: list[str]) -> list[str]:
    refs: list[str] = []
    for chunk in chunks:
        source_file = chunk.get("source_file")
        page = chunk.get("page")
        section_title = chunk.get("section_title")
        if source_file and page is not None:
            refs.append(f"{source_file}:{page}")
        elif source_file and section_title:
            refs.append(f"{source_file} / {section_title}")
        elif source_file:
            refs.append(str(source_file))
    refs.extend(str(source) for source in fallback_refs if source)
    return _dedupe(refs)


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
