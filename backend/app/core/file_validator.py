"""
File validation module for secure file upload handling.

Implements comprehensive validation including:
- File size limits
- MIME type validation
- Content type verification
- Filename sanitization
- Security checks
"""

import os
import re
import magic
from typing import Optional, Set
from fastapi import UploadFile, HTTPException
from dataclasses import dataclass
from ..core.logging_config import get_logger

logger = get_logger("core.file_validator")


@dataclass
class ValidationResult:
    """Result of file validation."""
    valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    sanitized_filename: Optional[str] = None
    file_size_mb: Optional[float] = None


class FileValidator:
    """
    Validates uploaded files for security and compliance.
    
    Validation steps:
    1. Check file size against limit
    2. Validate declared MIME type
    3. Verify file content matches declared type
    4. Sanitize filename to prevent path traversal
    """
    
    def __init__(
        self,
        max_size_mb: int = 10,
        allowed_types: Optional[Set[str]] = None
    ):
        """
        Initialize FileValidator.
        
        Args:
            max_size_mb: Maximum file size in megabytes (default: 10MB)
            allowed_types: Set of allowed MIME types (default: image types)
        """
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        if allowed_types is None:
            self.allowed_types = {
                "image/jpeg",
                "image/png",
                "image/webp"
            }
        else:
            self.allowed_types = allowed_types
        
        logger.info(
            f"FileValidator initialized: max_size={max_size_mb}MB, "
            f"allowed_types={self.allowed_types}"
        )
    
    async def validate_file(self, file: UploadFile) -> ValidationResult:
        """
        Validate uploaded file comprehensively.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            ValidationResult with validation status and details
            
        Raises:
            HTTPException: On validation failure with appropriate status code
        """
        try:
            # Read file content
            content = await file.read()
            file_size_bytes = len(content)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Reset file pointer for potential reuse
            await file.seek(0)
            
            # 1. Validate file size
            if file_size_bytes > self.max_size_bytes:
                logger.warning(
                    f"File size validation failed: {file_size_mb:.2f}MB exceeds "
                    f"{self.max_size_mb}MB limit (filename: {file.filename})"
                )
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error_code": "VAL_2002",
                        "message": f"File size exceeds maximum allowed size of {self.max_size_mb}MB",
                        "details": {
                            "file_size_mb": round(file_size_mb, 2),
                            "max_size_mb": self.max_size_mb,
                            "field": file.filename
                        }
                    }
                )
            
            # 2. Validate declared MIME type
            declared_type = file.content_type
            if declared_type not in self.allowed_types:
                logger.warning(
                    f"MIME type validation failed: {declared_type} not in allowed types "
                    f"(filename: {file.filename})"
                )
                raise HTTPException(
                    status_code=415,
                    detail={
                        "error_code": "VAL_2003",
                        "message": f"Unsupported media type: {declared_type}",
                        "details": {
                            "declared_type": declared_type,
                            "allowed_types": list(self.allowed_types),
                            "field": file.filename
                        }
                    }
                )
            
            # 3. Verify actual content type matches declared type
            actual_type = self._detect_content_type(content)
            if not self._types_match(declared_type, actual_type):
                logger.warning(
                    f"Content type mismatch: declared={declared_type}, "
                    f"actual={actual_type} (filename: {file.filename})"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "VAL_2004",
                        "message": "File content does not match declared type",
                        "details": {
                            "declared_type": declared_type,
                            "detected_type": actual_type,
                            "field": file.filename
                        }
                    }
                )
            
            # 4. Sanitize filename
            sanitized_name = self.sanitize_filename(file.filename or "upload")
            
            logger.info(
                f"File validation passed: {sanitized_name} "
                f"({file_size_mb:.2f}MB, {declared_type})"
            )
            
            return ValidationResult(
                valid=True,
                sanitized_filename=sanitized_name,
                file_size_mb=file_size_mb
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File validation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "SYS_9001",
                    "message": "Internal error during file validation"
                }
            )
    
    def _detect_content_type(self, content: bytes) -> str:
        """
        Detect actual content type from file content using magic numbers.
        
        Args:
            content: File content bytes
            
        Returns:
            Detected MIME type
        """
        try:
            mime = magic.Magic(mime=True)
            detected_type = mime.from_buffer(content)
            return detected_type
        except Exception as e:
            logger.error(f"Content type detection failed: {e}")
            return "application/octet-stream"
    
    def _types_match(self, declared: str, actual: str) -> bool:
        """
        Check if declared and actual MIME types match.
        
        Handles variations like image/jpg vs image/jpeg.
        
        Args:
            declared: Declared MIME type
            actual: Detected MIME type
            
        Returns:
            True if types match, False otherwise
        """
        # Normalize types
        declared = declared.lower().strip()
        actual = actual.lower().strip()
        
        # Direct match
        if declared == actual:
            return True
        
        # Handle jpeg variations
        jpeg_types = {"image/jpeg", "image/jpg"}
        if declared in jpeg_types and actual in jpeg_types:
            return True
        
        return False
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.
        
        Removes:
        - Path separators (/, \)
        - Parent directory references (..)
        - Special characters
        - Leading/trailing whitespace and dots
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for storage
        """
        if not filename:
            return "upload"
        
        # Get base filename (remove any path components)
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        # Keep only alphanumeric, dots, hyphens, underscores
        filename = re.sub(r'[^\w\s\-\.]', '', filename)
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove multiple consecutive dots (potential path traversal)
        filename = re.sub(r'\.{2,}', '.', filename)
        
        # Remove leading/trailing dots and whitespace
        filename = filename.strip('. \t\n\r')
        
        # Ensure filename is not empty after sanitization
        if not filename:
            filename = "upload"
        
        # Limit filename length (keep extension)
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            name = name[:max_length - len(ext)]
            filename = name + ext
        
        return filename
