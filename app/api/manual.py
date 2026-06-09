from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(tags=["manual"])


class ManualIndexRequest(BaseModel):
    doc_id: str = Field(..., description="Document id to index.")


@router.post("/manual/upload")
async def upload_manual(file: UploadFile = File(...)) -> dict[str, str | int | None]:
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "status": "received",
    }


@router.post("/manual/index")
async def index_manual(request: ManualIndexRequest) -> dict[str, str]:
    return {
        "doc_id": request.doc_id,
        "status": "index_pending",
        "message": "Indexing pipeline is not implemented yet.",
    }
