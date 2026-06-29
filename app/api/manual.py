from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.dependencies import get_manual_indexer, get_manual_store
from app.ingestion import InMemoryManualStore, ManualIndexer
from app.schemas.document import DocumentIndexResult, ManualUploadAndIndexResult, ManualUploadResult

router = APIRouter(tags=["manual"])


class ManualIndexRequest(BaseModel):
    doc_id: str = Field(..., description="Document id to index.")
    device_name: str | None = Field(default=None, description="Optional device name metadata.")
    device_model: str | None = Field(default=None, description="Optional device model metadata.")


@router.post("/manual/upload")
async def upload_manual(
    file: UploadFile = File(...),
    store: InMemoryManualStore = Depends(get_manual_store),
) -> ManualUploadResult:
    content = await file.read()
    filename = file.filename or "manual.txt"
    return store.save(filename=filename, content=content, content_type=file.content_type)


@router.post("/manual/upload-and-index")
async def upload_and_index_manual(
    file: UploadFile = File(...),
    store: InMemoryManualStore = Depends(get_manual_store),
    indexer: ManualIndexer = Depends(get_manual_indexer),
) -> ManualUploadAndIndexResult:
    content = await file.read()
    filename = file.filename or "manual.txt"
    upload_result = store.save(filename=filename, content=content, content_type=file.content_type)
    index_result = indexer.index_document(
        content=content,
        filename=filename,
        doc_id=upload_result.doc_id,
    )
    return ManualUploadAndIndexResult(
        doc_id=upload_result.doc_id,
        filename=upload_result.filename,
        content_type=upload_result.content_type,
        size=upload_result.size,
        upload_status=upload_result.status,
        index_status=index_result.status,
        chunks_count=index_result.chunks_count,
        content_type_stats=index_result.content_type_stats,
        sanitized_fields=index_result.sanitized_fields,
    )


@router.post("/manual/index")
async def index_manual(
    request: ManualIndexRequest,
    store: InMemoryManualStore = Depends(get_manual_store),
    indexer: ManualIndexer = Depends(get_manual_indexer),
) -> DocumentIndexResult:
    stored = store.get(request.doc_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Manual document not found: {request.doc_id}")
    return indexer.index_document(
        content=stored.content,
        filename=stored.filename,
        doc_id=stored.doc_id,
        device_name=request.device_name,
        device_model=request.device_model,
    )
