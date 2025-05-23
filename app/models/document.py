from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Document(BaseModel):
    """Document model for storing document metadata."""
    id: str
    filename: str
    upload_time: datetime
    content_type: str
    vector_store_id: Optional[str] = None

    class Config:
        from_attributes = True 