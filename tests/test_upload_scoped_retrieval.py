import asyncio
import json

from fastapi.testclient import TestClient

from app.core.dependencies import get_manual_indexer
from app.ingestion import ManualIndexer
from app.main import app
from app.mcp_server import MCPToolExecutor, create_default_tool_registry
from app.mcp_server import tools as mcp_tools


SCOPED_MANUAL = b"""# Fault Codes
Q77 Custom uploaded drive fault. Check the uploaded manual resolver cable.

# Safety Rules
Disconnect power before inspecting the uploaded drive.
"""

NO_MATCH_MANUAL = b"""# Overview
This uploaded manual only describes cabinet cleaning and label inspection.

# Maintenance
Wipe the housing with a dry cloth after shutdown.
"""

ACTION_MANUAL = """# 创建动作指令
创建动作指令时，先进入程序编辑界面。
1. 点击“新建程序”。
2. 选择“动作指令”。
3. 输入指令名称和目标位置。
4. 设置速度、等待时间和执行条件。
5. 点击“保存”并在空载状态下试运行。

# 补充说明
动作指令保存后，可在程序列表中查看和编辑。
""".encode("utf-8")

MANUAL_QA_QUALITY_MANUAL = """<!-- page: 10 -->
# 动作指令定义
动作指令是机器人程序中用于描述运动目标和执行条件的语句，由动作类型、位置、速度等信息构成。

<!-- page: 12 -->
# 创建动作指令步骤
1. 进入程序编辑界面。
2. 点击“增加指令”。
3. 选择“动作指令”。
4. 设置目标点、速度和等待条件。
5. 点击“确认”保存动作指令。

<!-- page: 13 -->
# 动作附加指令说明
动作附加指令用于补充速度、等待、输出等附加条件。
""".encode("utf-8")


def _sse_events(response_text: str) -> list[dict]:
    events: list[dict] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        events.append({"event": event_name, "data": json.loads(raw)})
        event_name = "message"
        data_lines = []

    for line in response_text.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
        elif not line.strip():
            flush()
    flush()
    return events


def _upload_and_index(client: TestClient, filename: str, content: bytes) -> dict:
    response = client.post(
        "/api/manual/upload-and-index",
        files={"file": (filename, content, "text/plain")},
    )
    assert response.status_code == 200
    return response.json()


def test_upload_and_index_indexes_document() -> None:
    client = TestClient(app)

    body = _upload_and_index(client, "uploaded-q77.txt", SCOPED_MANUAL)

    assert body["doc_id"]
    assert body["filename"] == "uploaded-q77.txt"
    assert body["upload_status"] == "received"
    assert body["index_status"] == "indexed"
    assert body["chunks_count"] > 0


def test_uploaded_document_question_uses_uploaded_source_refs() -> None:
    client = TestClient(app)
    upload = _upload_and_index(client, "uploaded-q77-source.txt", SCOPED_MANUAL)

    response = client.post(
        "/api/diagnosis/chat",
        json={
            "session_id": "test-uploaded-source",
            "message": "Q77 uploaded drive fault how to inspect?",
            "doc_id": upload["doc_id"],
            "doc_ids": [upload["doc_id"]],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    final_event = [event for event in events if event["event"] == "final_answer"][-1]
    source_refs = final_event["data"]["source_refs"]

    assert any("uploaded-q77-source.txt" in source for source in source_refs)
    assert all("mx100_manual.pdf" not in source for source in source_refs)


def test_uploaded_document_empty_result_does_not_fallback_to_demo() -> None:
    client = TestClient(app)
    upload = _upload_and_index(client, "uploaded-no-match.txt", NO_MATCH_MANUAL)

    response = client.post(
        "/api/diagnosis/chat",
        json={
            "session_id": "test-uploaded-no-match",
            "message": "E03 motor overheat how to repair?",
            "doc_id": upload["doc_id"],
            "doc_ids": [upload["doc_id"]],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    final_event = [event for event in events if event["event"] == "final_answer"][-1]

    assert "未在上传手册中找到依据" in final_event["data"]["final_answer"]
    assert all("mx100_manual.pdf" not in source for source in final_event["data"]["source_refs"])


def test_operation_manual_question_uses_manual_qa_answer() -> None:
    client = TestClient(app)
    upload = _upload_and_index(client, "action-command-manual.txt", ACTION_MANUAL)

    response = client.post(
        "/api/diagnosis/chat",
        json={
            "session_id": "test-manual-qa",
            "message": "怎么创建动作指令？",
            "doc_id": upload["doc_id"],
            "doc_ids": [upload["doc_id"]],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    diagnosis_event = [event for event in events if event["event"] == "diagnosis_completed"][-1]
    final_event = [event for event in events if event["event"] == "final_answer"][-1]
    final_answer = final_event["data"]["final_answer"]

    assert diagnosis_event["data"]["query_type"] == "manual_qa"
    assert "故障识别" not in final_answer
    assert "可能原因" not in final_answer
    assert "未识别到明确故障码" not in final_answer
    assert "选择“动作指令”" in final_answer
    assert "设置速度" in final_answer
    assert any("action-command-manual.txt" in source for source in final_event["data"]["source_refs"])
    assert all("mx100_manual.pdf" not in source for source in final_event["data"]["source_refs"])


def test_manual_qa_prefers_procedure_steps_over_definition() -> None:
    client = TestClient(app)
    upload = _upload_and_index(client, "manual-qa-quality.txt", MANUAL_QA_QUALITY_MANUAL)

    response = client.post(
        "/api/diagnosis/chat",
        json={
            "session_id": "test-manual-qa-quality",
            "message": "如何增加动作指令",
            "doc_id": upload["doc_id"],
            "doc_ids": [upload["doc_id"]],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    diagnosis_event = [event for event in events if event["event"] == "diagnosis_completed"][-1]
    final_event = [event for event in events if event["event"] == "final_answer"][-1]
    final_answer = final_event["data"]["final_answer"]
    source_refs = final_event["data"]["source_refs"]

    assert diagnosis_event["data"]["query_type"] == "manual_qa"
    assert "点击“增加指令”" in final_answer
    assert "选择“动作指令”" in final_answer
    assert "设置目标点" in final_answer
    assert "故障识别" not in final_answer
    assert "可能原因" not in final_answer
    assert "未识别到明确故障码" not in final_answer
    if "动作指令是机器人程序" in final_answer:
        assert final_answer.index("点击“增加指令”") < final_answer.index("动作指令是机器人程序")
    assert source_refs
    assert all(source.startswith("manual-qa-quality.txt:") for source in source_refs)
    assert "manual-qa-quality.txt" not in source_refs
    assert all("mx100_manual.pdf" not in source for source in source_refs)


def test_without_uploaded_document_demo_fallback_is_preserved(monkeypatch) -> None:
    empty_indexer = ManualIndexer()
    monkeypatch.setattr(mcp_tools, "get_manual_indexer", lambda: empty_indexer)

    result = asyncio.run(
        MCPToolExecutor(create_default_tool_registry()).execute_tool(
            "manual_hybrid_search",
            {
                "query": "E03 motor overheat",
                "device_model": "MX100",
                "content_types": ["fault_code"],
                "top_k_bm25": 3,
                "top_k_dense": 3,
                "top_n_rerank": 3,
            },
        )
    )

    assert result.status == "success"
    assert result.data is not None
    assert any("mx100_manual.pdf" in source for source in result.data["source_refs"])


def test_mcp_tool_with_doc_ids_does_not_fallback_to_demo() -> None:
    indexer = get_manual_indexer()
    index_result = indexer.index_document(
        NO_MATCH_MANUAL,
        filename="scoped-no-demo.txt",
        doc_id="scoped-no-demo-doc",
    )
    assert index_result.status == "indexed"

    result = asyncio.run(
        MCPToolExecutor(create_default_tool_registry()).execute_tool(
            "manual_hybrid_search",
            {
                "query": "E03 motor overheat",
                "doc_ids": ["scoped-no-demo-doc"],
                "content_types": ["fault_code"],
                "top_k_bm25": 3,
                "top_k_dense": 3,
                "top_n_rerank": 3,
            },
        )
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["results"] == []
    assert result.data["source_refs"] == []
