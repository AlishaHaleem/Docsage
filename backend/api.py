from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.dependencies import get_rag
from app.rag_pipeline import RAG
from fastapi.responses import JSONResponse
import os
import shutil

router = APIRouter()

@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    rag: RAG = Depends(get_rag)
):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".txt"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc_type = "pdf" if ext == ".pdf" else "txt"
        rag.process_and_store_documents([temp_path], doc_type=doc_type)
        return {"message": "File processed and stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_rag(
    request: dict,
    rag: RAG = Depends(get_rag)
):
    try:
        question = request.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        answer = rag.generate_answer(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
