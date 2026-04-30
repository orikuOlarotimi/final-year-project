from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
import os
from datetime import datetime
from uuid import uuid4
from bson import ObjectId
from app.models.document import DocumentModel
from app.models.chat import Chat
from app.core.dependencies import get_current_user
from app.services.document_processor import process_document


router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    chat_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    try:
        if not chat_id or chat_id.strip() == "":
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "chat_id is required"}
            )

        chat_id = chat_id.strip()
        if not ObjectId.is_valid(chat_id):
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "Invalid chat_id format"}
            )
        # 🔹 1. Validate chat ownership
        chat = await Chat.find_one(
            Chat.id == ObjectId(chat_id),
            Chat.user_id == user_id
        )

        if not chat:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "Chat not found"}
            )

        # 🔹 2. Validate file presence
        if not file:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "File is required"}
            )

        # 🔹 3. Validate file extension
        filename = file.filename
        ext = filename.split(".")[-1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "Invalid file type"}
            )

        # 🔹 4. Read file (to check size)
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "File too large"}
            )

        # 🔹 5. Generate document ID
        temp_id = str(uuid4())

        # 🔹 6. Create file path
        upload_dir = f"uploads/{user_id}/{chat_id}"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = f"{upload_dir}/{temp_id}.{ext}"

        # 🔹 7. Save file
        with open(file_path, "wb") as f:
            f.write(contents)

        # 🔹 8. Create document record
        document = DocumentModel(
            user_id=user_id,
            chat_id=chat_id,
            filename=filename,
            file_path=file_path,
            file_type=ext,
            file_size=len(contents),
            status="uploaded",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()

        )

        await document.insert()

        background_tasks.add_task(process_document, str(document.id))

        return {
            "success": True,
            "message": "File uploaded successfully",
            "document": {
                "id": str(document.id),
                "filename": filename,
                "status": "uploaded"
            }
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Upload failed"}
        )