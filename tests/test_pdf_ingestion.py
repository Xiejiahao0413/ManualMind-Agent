import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ingestion import ManualIndexer, PDFManualParser, TextManualParser, UnsupportedPDFTypeError


PDF_MANUALS_DIR = Path("data/pdf_manuals")


BLANK_TEXT_BASED_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
184
%%EOF
"""


def test_pdf_parser_can_parse_text_based_pdf() -> None:
    parser = PDFManualParser()
    pdf_path = PDF_MANUALS_DIR / "pdf_a100_air_compressor_manual.pdf"

    document = parser.parse(
        pdf_path.read_bytes(),
        filename=pdf_path.name,
        doc_id="pdf-a100-test",
        device_name="Air Compressor A100",
        device_model="A100",
    )

    assert document.doc_id == "pdf-a100-test"
    assert "Synthetic demo data. Not a real manufacturer manual." in document.text
    assert "E03" in document.text
    assert "<!-- page: 1 -->" in document.text
    assert document.metadata["pages_count"] == 2


def test_pdf_chunks_include_page_metadata() -> None:
    indexer = ManualIndexer()
    pdf_path = PDF_MANUALS_DIR / "pdf_a100_air_compressor_manual.pdf"

    result = indexer.index_document(
        pdf_path.read_bytes(),
        filename=pdf_path.name,
        doc_id="pdf-a100-pages",
        device_name="Air Compressor A100",
        device_model="A100",
    )
    chunks = [chunk for chunk in indexer.list_chunks() if chunk.doc_id == "pdf-a100-pages"]

    assert result.status == "indexed"
    assert chunks
    assert {chunk.page for chunk in chunks if chunk.page} >= {1, 2}
    assert all(chunk.source_file == pdf_path.name for chunk in chunks)


def test_empty_pdf_returns_clear_unsupported_error() -> None:
    parser = PDFManualParser()

    with pytest.raises(UnsupportedPDFTypeError, match="unsupported_pdf_type"):
        parser.parse(BLANK_TEXT_BASED_PDF, filename="blank.pdf", doc_id="blank-pdf")


def test_markdown_ingestion_still_uses_text_parser() -> None:
    parser = TextManualParser()

    document = parser.parse(b"# Fault Codes\nE03 overheat", filename="manual.md", doc_id="md-still-ok")

    assert document.doc_id == "md-still-ok"
    assert "E03" in document.text
    assert document.metadata["extension"] == ".md"


def test_index_pdf_manuals_script_runs() -> None:
    output = _run_script("scripts/index_pdf_manuals.py")

    assert output["indexed_docs_count"] >= 2
    assert output["total_chunks_count"] >= 6
    assert output["pages_count"] >= 4
    assert output["unsupported_files"] == []
    assert len(output["source_files"]) >= 2


def test_demo_pdf_manuals_script_runs() -> None:
    output = _run_script("scripts/demo_pdf_manuals.py")

    assert output["index_summary"]["indexed_docs_count"] >= 2
    assert len(output["results"]) == 2
    for item in output["results"]:
        assert item["trace_id"]
        assert item["source_refs"]
        assert item["final_answer"]
        assert item["pages"]


def _run_script(script_path: str) -> dict:
    result = subprocess.run(
        [sys.executable, script_path],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(result.stdout)
