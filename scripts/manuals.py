from pathlib import Path
from typing import TypedDict

from app.core.dependencies import get_manual_indexer
from app.ingestion import ManualIndexer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = PROJECT_ROOT / "data" / "manuals"


class IndexedManualSummary(TypedDict):
    indexed_docs_count: int
    total_chunks_count: int
    fault_codes_count: int
    device_models: list[str]


DEVICE_CATALOG: dict[str, tuple[str, str]] = {
    "a100_air_compressor.md": ("空压机 A100", "A100"),
    "b200_hydraulic_press.md": ("液压机 B200", "B200"),
    "c300_conveyor_motor.md": ("输送电机 C300", "C300"),
    "d400_cooling_system.md": ("冷却系统 D400", "D400"),
    "g600_sensor_gateway.md": ("传感器网关 G600", "G600"),
    "h800_robot_arm.md": ("机械臂 H800", "H800"),
    "k900_packaging_machine.md": ("包装机 K900", "K900"),
    "p500_power_module.md": ("电源模块 P500", "P500"),
    "t100_temperature_controller.md": ("温控器 T100", "T100"),
    "w700_pump_controller.md": ("水泵控制器 W700", "W700"),
}


def index_manuals(
    manuals_dir: Path = MANUALS_DIR,
    indexer: ManualIndexer | None = None,
) -> IndexedManualSummary:
    active_indexer = indexer or get_manual_indexer()
    manual_paths = sorted(manuals_dir.glob("*.md"))
    indexed_docs_count = 0

    for path in manual_paths:
        device_name, device_model = infer_device_metadata(path)
        active_indexer.index_document(
            content=path.read_bytes(),
            filename=path.name,
            doc_id=path.stem,
            device_name=device_name,
            device_model=device_model,
        )
        indexed_docs_count += 1

    chunks = active_indexer.list_chunks()
    device_models = sorted({chunk.device_model for chunk in chunks if chunk.device_model})
    fault_codes_count = sum(1 for chunk in chunks if chunk.fault_code)
    return {
        "indexed_docs_count": indexed_docs_count,
        "total_chunks_count": len(chunks),
        "fault_codes_count": fault_codes_count,
        "device_models": device_models,
    }


def infer_device_metadata(path: Path) -> tuple[str, str]:
    catalog_entry = DEVICE_CATALOG.get(path.name)
    if catalog_entry:
        return catalog_entry
    model = path.stem.split("_", maxsplit=1)[0].upper()
    return model, model
