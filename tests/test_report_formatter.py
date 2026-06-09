from app.agents.reporting import build_diagnosis_report
from app.schemas.diagnosis import DiagnosisState


E03_DESCRIPTION = """E03 表示温度传感器异常。控制器检测到温度传感器信号超出正常范围。

可能原因：
- 温度传感器接线松动或端子氧化。
- 温度传感器接线松动或端子氧化。
- 冷却风扇堵塞导致局部温度异常。

排查步骤：
1. 先断电并等待设备冷却。
2. 检查温度传感器插头、线束和端子。
2. 检查温度传感器插头、线束和端子。

安全提醒：处理 E03 前必须断电、等待冷却，禁止带压拆卸温度传感器。
"""


def test_report_formatter_outputs_chinese_sections_without_english_templates() -> None:
    report = build_diagnosis_report(_sample_state())

    for section in ("故障识别", "可能原因", "排查步骤", "安全提醒", "引用来源"):
        assert section in report
    assert "Review retrieved manual evidence" not in report
    assert "Check the fault code or parameter range against the cited source" not in report
    assert "No high-risk operation detected by current tool results" not in report


def test_report_formatter_does_not_repeat_full_e03_description() -> None:
    report = build_diagnosis_report(_sample_state())

    assert report.count("E03 表示温度传感器异常") == 1
    assert report.count("温度传感器接线松动或端子氧化") == 1
    assert report.count("检查温度传感器插头、线束和端子") == 1


def test_report_formatter_deduplicates_source_refs() -> None:
    report = build_diagnosis_report(_sample_state())

    assert report.count("- a100_manual.md:1") == 1


def _sample_state() -> DiagnosisState:
    return DiagnosisState(
        task_id="task-report",
        session_id="session-report",
        user_query="空压机 A100 报 E03",
        fault_code="E03",
        query_type="fault_code",
        risk_level="medium",
        fault_info={
            "status": "found",
            "fault_code": "E03",
            "description": E03_DESCRIPTION,
            "source_refs": ["a100_manual.md:1"],
        },
        source_refs=["a100_manual.md:1", "a100_manual.md:1"],
    )
