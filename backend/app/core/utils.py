import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
import uuid
from typing import Optional

UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")

# File size limit: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

def save_upload_file(upload_file: UploadFile, directory: str = "uploads") -> str:
    """
    Saves an uploaded file to disk and returns the absolute path.
    
    Args:
        upload_file: The uploaded file object
        directory: Subdirectory name under data/ (default: "uploads")
    
    Returns:
        Absolute path to saved file
        
    Raises:
        HTTPException: If file validation fails
    """
    # Validate file type
    if upload_file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {upload_file.content_type}. "
                   f"Allowed types: {', '.join(ALLOWED_TYPES)}"
        )
    
    # Create directory
    save_dir = Path("data") / directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    extension = Path(upload_file.filename).suffix if upload_file.filename else ".jpg"
    file_path = save_dir / f"{file_id}{extension}"
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    return str(file_path.absolute())


def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours
    
    Returns:
        Number of files deleted
    """
    import time
    
    clean_dir = Path("data") / directory
    if not clean_dir.exists():
        return 0
    
    deleted_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in clean_dir.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception:
                    pass  # Skip files that can't be deleted
    
    return deleted_count

