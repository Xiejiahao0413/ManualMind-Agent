import json
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000").strip() or "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120


def build_endpoint_candidates(base_url: str, path: str) -> list[str]:
    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    if normalized_path.startswith("/api/"):
        return [f"{normalized_base}{normalized_path}"]
    return [
        f"{normalized_base}{normalized_path}",
        f"{normalized_base}/api{normalized_path}",
    ]


def post_with_api_fallback(
    base_url: str,
    path: str,
    **kwargs: Any,
) -> tuple[requests.Response, str]:
    last_error: requests.RequestException | None = None
    last_response: requests.Response | None = None

    for url in build_endpoint_candidates(base_url, path):
        try:
            response = requests.post(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            continue

        if response.status_code != 404:
            return response, url

        last_response = response
        response.close()

    if last_response is not None:
        return last_response, build_endpoint_candidates(base_url, path)[-1]

    if last_error is not None:
        raise last_error

    raise RuntimeError("No endpoint candidate was attempted.")


def response_text_preview(response: requests.Response, limit: int = 1000) -> str:
    return response.text[:limit]


def parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_event: str | None = None
    data_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_event, data_lines
        if current_event is None and not data_lines:
            return

        raw_data = "\n".join(data_lines).strip()
        parsed_data: Any = raw_data
        if raw_data:
            try:
                parsed_data = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed_data = {"raw": raw_data}

        events.append(
            {
                "event": current_event or "message",
                "data": parsed_data,
            }
        )
        current_event = None
        data_lines = []

    for line in raw_text.splitlines():
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
        elif not line.strip():
            flush_current()

    flush_current()
    return events


def build_diagnosis_message(query: str, device_model: str) -> str:
    clean_query = query.strip()
    clean_model = device_model.strip()
    if clean_model and clean_model not in clean_query:
        return f"{clean_model} {clean_query}"
    return clean_query


st.set_page_config(
    page_title="ManualMind-Agent",
    layout="wide",
)

st.title("ManualMind-Agent")
st.caption("Multi-Agent Equipment Fault Diagnosis System Demo")

backend_url = st.sidebar.text_input(
    "FastAPI backend URL",
    value=DEFAULT_BACKEND_URL,
    help="The UI uses /api/manual/upload-and-index and /api/diagnosis/chat for the current backend.",
)

st.info(
    "Synthetic demo data only. This UI is a lightweight product demo wrapper for the existing FastAPI backend."
)

upload_result: dict[str, Any] | None = None
diagnosis_result: dict[str, Any] | None = None
diagnosis_events: list[dict[str, Any]] = []
request_url: str | None = None

if "current_doc_id" not in st.session_state:
    st.session_state["current_doc_id"] = None
if "current_filename" not in st.session_state:
    st.session_state["current_filename"] = None
if "chunks_count" not in st.session_state:
    st.session_state["chunks_count"] = 0
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False

upload_section, qa_section = st.columns(2)

with upload_section:
    st.header("Manual Upload")
    uploaded_file = st.file_uploader(
        "Upload a manual file",
        type=["pdf", "txt"],
        help="Supported UI upload types: PDF and TXT.",
    )

    if st.button("Upload", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.error("Please select a PDF or TXT file first.")
        else:
            upload_url = f"{backend_url.rstrip('/')}/api/manual/upload-and-index"
            upload_url_candidates = [upload_url]
            st.caption(f"Upload URL: {upload_url}")
            with st.expander("Upload debug", expanded=False):
                st.write("Upload URL candidates")
                st.code("\n".join(upload_url_candidates))
            with st.spinner("Uploading manual..."):
                try:
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.getvalue()
                    files = {
                        "file": (
                            uploaded_file.name,
                            file_bytes,
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    request_url = upload_url
                    response = requests.post(
                        upload_url,
                        files=files,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    if response.ok:
                        upload_result = response.json()
                        st.session_state["current_doc_id"] = upload_result.get("doc_id")
                        st.session_state["current_filename"] = upload_result.get("filename")
                        st.session_state["chunks_count"] = upload_result.get("chunks_count", 0)
                        st.session_state["indexed"] = upload_result.get("index_status") == "indexed"
                        st.success(f"Upload and index succeeded via {request_url}")
                    else:
                        st.error("Upload failed.")
                        st.json(
                            {
                                "upload_url": request_url,
                                "status_code": response.status_code,
                                "response_text_preview": response_text_preview(response),
                            }
                        )
                except requests.RequestException as exc:
                    st.error("Upload request failed.")
                    st.json(
                        {
                            "upload_url": upload_url,
                            "upload_url_candidates": upload_url_candidates,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        }
                    )
                except ValueError:
                    st.error("Upload response was not valid JSON.")
                    if request_url is not None:
                        st.json(
                            {
                                "upload_url": request_url,
                                "status_code": response.status_code,
                                "response_text_preview": response_text_preview(response),
                            }
                        )

with qa_section:
    st.header("Diagnosis Chat")
    if st.session_state["indexed"]:
        st.success(
            f"Using uploaded manual: {st.session_state['current_filename']} "
            f"(doc_id={st.session_state['current_doc_id']}, chunks={st.session_state['chunks_count']})"
        )
    else:
        st.warning("No uploaded manual selected. Questions will use the built-in demo fallback.")

    query = st.text_area(
        "query",
        value="A100 显示 E03，应该如何排查？",
        height=120,
    )
    device_model = st.text_input("device_model", value="A100")

    if st.button("Send", type="primary"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            with st.spinner("Running multi-agent diagnosis workflow..."):
                try:
                    payload = {
                        "session_id": "streamlit-demo-session",
                        "message": build_diagnosis_message(query, device_model),
                        "stream": True,
                    }
                    if st.session_state["current_doc_id"]:
                        payload["doc_id"] = st.session_state["current_doc_id"]
                        payload["doc_ids"] = [st.session_state["current_doc_id"]]
                    response, request_url = post_with_api_fallback(
                        backend_url,
                        "/api/diagnosis/chat",
                        json=payload,
                    )
                    if not response.ok:
                        st.error(f"Diagnosis request failed: {response.status_code} {response.text}")
                    else:
                        content_type = response.headers.get("content-type", "")
                        if "text/event-stream" in content_type:
                            diagnosis_events = parse_sse_events(response.text)
                            diagnosis_result = {
                                "request_url": request_url,
                                "events": diagnosis_events,
                            }
                        else:
                            diagnosis_result = response.json()
                            diagnosis_events = []
                        st.success(f"Diagnosis request completed via {request_url}")
                except requests.RequestException as exc:
                    st.error(f"Diagnosis request failed: {exc}")
                except ValueError:
                    st.error("Diagnosis response was not valid JSON or SSE data.")

st.header("Result")

if upload_result is not None:
    st.subheader("Upload JSON")
    st.json(upload_result)

st.subheader("Current Manual Scope")
st.json(
    {
        "indexed": st.session_state["indexed"],
        "doc_id": st.session_state["current_doc_id"],
        "filename": st.session_state["current_filename"],
        "chunks_count": st.session_state["chunks_count"],
    }
)

if diagnosis_result is not None:
    final_events = [
        event
        for event in diagnosis_events
        if event.get("event") in {"final_answer", "handoff_required"}
    ]
    if final_events:
        st.subheader("Final Output")
        st.json(final_events[-1])

        final_data = final_events[-1].get("data")
        if isinstance(final_data, dict) and final_data.get("final_answer"):
            st.markdown(final_data["final_answer"])
        if isinstance(final_data, dict) and final_data.get("retrieval_debug"):
            st.subheader("Retrieval Debug")
            st.json(final_data["retrieval_debug"])

    st.subheader("Structured Response")
    st.json(diagnosis_result)
else:
    st.write("Upload a manual or send a diagnosis query to see results here.")
