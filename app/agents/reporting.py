import re
from typing import Any

from app.mcp_server.tools import normalize_source_refs
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


PROCEDURE_TOPIC_TERMS = ("动作指令", "指令", "程序", "参数", "action instruction", "instruction")
PROCEDURE_ACTION_TERMS = (
    "创建",
    "新建",
    "增加",
    "添加",
    "插入",
    "示教",
    "步骤",
    "点击",
    "选择",
    "进入",
    "设置",
    "确认",
    "保存",
    "create",
    "add",
    "insert",
    "teach",
    "click",
    "select",
    "enter",
    "set",
    "confirm",
    "save",
)
DEFINITION_TERMS = ("定义", "构成", "是指", "如下信息构成", "包括以下信息", "consists of", "is defined")
NOISE_LINE_PATTERN = re.compile(
    r"^(?:图\s*\d+(?:\.\d+)*|图|如下图所示|如图\s*\d+(?:\.\d+)*\s*所示|\d+\s*/\s*\d+|第\s*\d+\s*页|page\s*\d+)$",
    re.IGNORECASE,
)


def build_manual_qa_answer(state: DiagnosisState) -> str:
    chunks = sorted(
        [
            chunk
            for chunk in state.retrieved_chunks
            if _clean_manual_qa_text(chunk.get("text") or "")
        ],
        key=lambda chunk: _manual_qa_chunk_score(state, chunk),
        reverse=True,
    )
    if not chunks:
        return "未在上传手册中找到依据。请确认上传的手册是否已完成索引，或补充更具体的章节、功能名称后重试。"

    primary_chunk = chunks[0]
    operation_steps = _extract_operation_items(primary_chunk.get("text") or "")
    related_items: list[str] = []
    for chunk in chunks[1:3]:
        related_items.extend(_extract_relevant_manual_qa_items(chunk.get("text") or "")[:3])

    if operation_steps:
        heading = "根据上传手册，操作步骤如下："
        answer_items = operation_steps[:8]
    else:
        heading = "手册中未检索到完整创建步骤，但找到相关说明："
        answer_items = _extract_relevant_manual_qa_items(primary_chunk.get("text") or "")[:5]

    source_refs = _manual_qa_source_refs(chunks, state.source_refs)
    sections = [heading + "\n\n" + _format_numbered(answer_items)]
    if related_items:
        sections.append("补充说明：\n" + _format_bullets(related_items[:5]))
    sections.append("引用来源：\n" + _format_bullets(source_refs or ["暂无引用来源"]))
    return "\n\n".join(sections)


def _manual_qa_chunk_score(state: DiagnosisState, chunk: dict) -> float:
    query = (state.raw_query or state.sanitized_query or state.user_query or "").lower()
    combined = f"{chunk.get('section_title') or ''}\n{chunk.get('text') or ''}".lower()
    score = float(chunk.get("rerank_score") or chunk.get("score") or 0.0)
    query_terms = _manual_qa_query_terms(query)
    score += sum(0.4 for term in query_terms if term and term in combined)
    topic_hits = sum(1 for term in PROCEDURE_TOPIC_TERMS if term.lower() in combined)
    action_hits = sum(1 for term in PROCEDURE_ACTION_TERMS if term.lower() in combined)
    definition_hits = sum(1 for term in DEFINITION_TERMS if term.lower() in combined)
    if topic_hits and action_hits:
        score += 5.0
    if topic_hits and action_hits >= 2:
        score += 2.0
    if _has_numbered_steps(chunk.get("text") or ""):
        score += 2.0
    if definition_hits and action_hits == 0:
        score -= 4.0
    elif definition_hits:
        score -= 1.0
    return score


def _manual_qa_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z]+\d+|\d+[a-z]+|[a-z0-9_-]+|[\u4e00-\u9fff]{2,}", query)
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", term):
            expanded.extend(term[index : index + 2] for index in range(len(term) - 1))
    return expanded


def _extract_operation_items(text: Any) -> list[str]:
    items: list[str] = []
    for item in _extract_relevant_manual_qa_items(text):
        if _looks_like_operation_step(item):
            items.append(item)
    return _dedupe(items)


def _extract_relevant_manual_qa_items(text: Any) -> list[str]:
    cleaned_text = _clean_manual_qa_text(text)
    candidates = _split_items(cleaned_text)
    if not candidates:
        candidates = _split_sentences(cleaned_text)
    return _dedupe(item for item in candidates if _is_useful_manual_qa_item(item))


def _clean_manual_qa_text(text: Any) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for line in normalized.splitlines():
        cleaned = _clean_text(line)
        cleaned = re.sub(r"图\s*\d+(?:\.\d+)*", "", cleaned).strip()
        cleaned = cleaned.replace("如下图所示", "").replace("如图所示", "").strip()
        if not cleaned or NOISE_LINE_PATTERN.match(cleaned):
            continue
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def _is_useful_manual_qa_item(item: str) -> bool:
    cleaned = _clean_text(item)
    if len(cleaned) < 4:
        return False
    if NOISE_LINE_PATTERN.match(cleaned):
        return False
    return True


def _looks_like_operation_step(text: str) -> bool:
    lower_text = text.lower()
    return any(term.lower() in lower_text for term in PROCEDURE_ACTION_TERMS)


def _is_definition_like(text: Any) -> bool:
    lower_text = str(text).lower()
    return any(term.lower() in lower_text for term in DEFINITION_TERMS)


def _has_numbered_steps(text: Any) -> bool:
    return bool(re.search(r"(^|\n)\s*(?:\d+[.、)]|[（(]\d+[）)])", str(text)))


def _manual_qa_source_refs(chunks: list[dict], fallback_refs: list[str]) -> list[str]:
    refs: list[str] = []
    for chunk in chunks:
        source_file = chunk.get("source_file")
        page = chunk.get("page")
        section_title = chunk.get("section_title")
        if source_file and page is not None:
            refs.append(f"{source_file}:{page}")
        elif source_file and section_title and not _is_page_number_title(str(section_title)):
            refs.append(f"{source_file} / {section_title}")
        elif source_file:
            refs.append(str(source_file))
    refs.extend(str(source) for source in fallback_refs if source)
    return normalize_source_refs(refs)


# Final manual QA formatter override. Keep this scoped to manual_qa so fault diagnosis
# reports continue to use the original diagnosis formatter above.
MANUAL_QA_ACTION_TERMS = (
    "点击",
    "选择",
    "进入",
    "设置",
    "创建",
    "新建",
    "增加",
    "添加",
    "插入",
    "示教",
    "确认",
    "保存",
    "修改",
    "选中",
    "click",
    "select",
    "enter",
    "set",
    "create",
    "add",
    "insert",
    "teach",
    "confirm",
    "save",
)
MANUAL_QA_DEFINITION_TERMS = ("定义", "构成", "是指", "如下信息构成", "包括以下信息", "用于描述")
MANUAL_QA_NOISE_TITLES = {"指令选择窗口", "笔型光标", "程序编辑语句界面"}


def build_manual_qa_answer(state: DiagnosisState) -> str:
    chunks = sorted(
        [
            chunk
            for chunk in state.retrieved_chunks
            if _manual_qa_clean_text(chunk.get("text") or "")
        ],
        key=lambda chunk: _manual_qa_chunk_score_v2(state, chunk),
        reverse=True,
    )
    if not chunks:
        return "未在上传手册中找到依据。请确认上传的手册是否已完成索引，或补充更具体的章节、功能名称后重试。"

    primary_chunk = chunks[0]
    steps = _manual_qa_extract_steps(primary_chunk.get("text") or "")
    used_chunks = [primary_chunk]
    notes: list[str] = []

    if steps:
        heading = "根据上传手册，操作步骤如下："
        answer_items = steps[:8]
        if len(answer_items) < 3:
            for chunk in chunks[1:2]:
                extra_steps = _manual_qa_extract_steps(chunk.get("text") or "")
                if extra_steps:
                    answer_items.extend(extra_steps[: 4 - len(answer_items)])
                    used_chunks.append(chunk)
                    break
    else:
        heading = "手册中未检索到完整创建步骤，但找到相关说明："
        answer_items = _manual_qa_extract_related_items(primary_chunk.get("text") or "")[:5]

    if steps:
        for chunk in chunks[1:2]:
            related = _manual_qa_extract_related_items(chunk.get("text") or "")
            related = [item for item in related if not _manual_qa_is_definition_like(item)]
            if related:
                notes = related[:2]
                used_chunks.append(chunk)
                break

    source_refs = _manual_qa_source_refs_v2(used_chunks, state.source_refs if len(used_chunks) > 1 else [])
    sections = [heading + "\n\n" + _format_numbered(answer_items)]
    if notes:
        sections.append("补充说明：\n" + _format_bullets(notes[:2]))
    sections.append("引用来源：\n" + _format_bullets(source_refs or ["暂无引用来源"]))
    return "\n\n".join(sections)


def _manual_qa_chunk_score_v2(state: DiagnosisState, chunk: dict) -> float:
    text = _manual_qa_clean_text(f"{chunk.get('section_title') or ''}\n{chunk.get('text') or ''}")
    query = (state.raw_query or state.sanitized_query or state.user_query or "").lower()
    score = float(chunk.get("rerank_score") or chunk.get("score") or 0.0)
    score += sum(0.5 for term in _manual_qa_query_terms(query) if term and term in text.lower())
    action_hits = sum(1 for term in MANUAL_QA_ACTION_TERMS if term.lower() in text.lower())
    definition_hits = sum(1 for term in MANUAL_QA_DEFINITION_TERMS if term.lower() in text.lower())
    if action_hits:
        score += 4.0 + min(action_hits, 4)
    if _manual_qa_has_numbered_steps(text):
        score += 2.0
    if definition_hits and action_hits == 0:
        score -= 5.0
    elif definition_hits:
        score -= 1.0
    return score


def _manual_qa_clean_text(text: Any) -> str:
    merged = _manual_qa_merge_pdf_lines(str(text))
    cleaned_lines: list[str] = []
    for raw_line in merged.splitlines():
        line = _clean_text(raw_line)
        line = re.sub(r"图\s*\d+(?:\.\d+)*\s*[\u4e00-\u9fffA-Za-z0-9_-]*", "", line).strip()
        line = re.sub(r"如\s*(?:下)?图\s*所示", "", line).strip()
        line = re.sub(r"如\s*所示", "", line).strip()
        if not line or _manual_qa_is_noise_line(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _manual_qa_merge_pdf_lines(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    merged: list[str] = []
    sentence_end = tuple("。！？!?；;：:")
    list_start_pattern = re.compile(r"^(?:\d+[.、)]|[（(]\d+[）)]|[-*])\s*")

    for line in lines:
        if not line:
            continue
        if not merged:
            merged.append(line)
            continue
        previous = merged[-1]
        if list_start_pattern.match(line):
            merged.append(line)
            continue
        if previous.endswith(sentence_end):
            merged.append(line)
            continue
        if _manual_qa_is_noise_line(previous) or _manual_qa_is_noise_line(line):
            merged.append(line)
            continue
        joiner = "" if _manual_qa_should_join_without_space(previous, line) else " "
        merged[-1] = previous + joiner + line
    return "\n".join(merged)


def _manual_qa_should_join_without_space(previous: str, current: str) -> bool:
    if not previous or not current:
        return True
    if "\u4e00" <= previous[-1] <= "\u9fff" or "\u4e00" <= current[0] <= "\u9fff":
        return True
    if previous[-1].isalnum() and current[0].isalnum() and (len(current) <= 3 or len(previous) <= 3):
        return True
    return False


def _manual_qa_extract_steps(text: Any) -> list[str]:
    cleaned = _manual_qa_clean_text(text)
    candidates = _split_items(cleaned) or _split_sentences(cleaned)
    steps: list[str] = []
    for candidate in candidates:
        item = _manual_qa_normalize_item(candidate)
        if not item or _manual_qa_is_step_heading(item) or _manual_qa_is_noise_line(item):
            continue
        if _manual_qa_is_operation_step(item):
            steps.append(item)
    return _dedupe(steps)


def _manual_qa_extract_related_items(text: Any) -> list[str]:
    cleaned = _manual_qa_clean_text(text)
    candidates = _split_items(cleaned) or _split_sentences(cleaned)
    result: list[str] = []
    for candidate in candidates:
        item = _manual_qa_normalize_item(candidate)
        if not item or _manual_qa_is_step_heading(item) or _manual_qa_is_noise_line(item):
            continue
        if len(item) >= 6:
            result.append(item)
    return _dedupe(result)


def _manual_qa_normalize_item(text: str) -> str:
    item = _clean_text(text)
    item = re.sub(r"^\s*(?:\d+[.、)]|[（(]\d+[）)]|[-*])\s*", "", item)
    item = re.sub(r"图\s*\d+(?:\.\d+)*", "", item).strip()
    item = re.sub(r"如\s*所示", "", item).strip()
    return item


def _manual_qa_is_operation_step(text: str) -> bool:
    lower_text = text.lower()
    return any(term.lower() in lower_text for term in MANUAL_QA_ACTION_TERMS)


def _manual_qa_is_definition_like(text: str) -> bool:
    lower_text = text.lower()
    return any(term.lower() in lower_text for term in MANUAL_QA_DEFINITION_TERMS)


def _manual_qa_is_step_heading(text: str) -> bool:
    cleaned = _clean_text(text)
    return bool(re.search(r"(步骤如下|操作步骤|方法如下|流程如下)[:：]?$", cleaned))


def _manual_qa_is_noise_line(text: str) -> bool:
    cleaned = _clean_text(text)
    if cleaned in MANUAL_QA_NOISE_TITLES:
        return True
    if re.match(r"^(?:图\s*\d+(?:\.\d+)*.*|图|指令选择窗口|\d+\s*/\s*\d+|第\s*\d+\s*页|page\s*\d+)$", cleaned, re.IGNORECASE):
        return True
    return cleaned in {"如所示", "如下图所示", "如图所示"}


def _manual_qa_has_numbered_steps(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*(?:\d+[.、)]|[（(]\d+[）)])", text))


def _manual_qa_source_refs_v2(chunks: list[dict], fallback_refs: list[str]) -> list[str]:
    refs: list[str] = []
    for chunk in chunks:
        source_file = chunk.get("source_file")
        page = chunk.get("page")
        section_title = chunk.get("section_title")
        if source_file and page is not None:
            refs.append(f"{source_file}:{page}")
        elif source_file and section_title and not _manual_qa_is_page_title(str(section_title)):
            refs.append(f"{source_file} / {section_title}")
        elif source_file:
            refs.append(str(source_file))
    refs.extend(str(ref) for ref in fallback_refs if ref)
    return normalize_source_refs(refs)


def _manual_qa_is_page_title(title: str) -> bool:
    return bool(re.match(r"^(?:\d+\s*/\s*\d+|\d+|第\s*\d+\s*页|page\s*\d+)(?:\s*/\s*\d+)?$", title.strip(), re.IGNORECASE))


def _is_page_number_title(title: str) -> bool:
    return bool(
        re.match(
            r"^(?:\d+\s*/\s*\d+|\d+|第\s*\d+\s*页|page\s*\d+)(?:\s*/\s*\d+)?$",
            title.strip(),
            re.IGNORECASE,
        )
    )
