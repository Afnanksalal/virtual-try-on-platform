"""
Supabase Storage Service - ALL file operations go through Supabase.
NO local file storage allowed.
"""

import os
import io
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional, BinaryIO, List, Dict, Any, Callable
from functools import wraps
from PIL import Image
from fastapi import UploadFile
from supabase import create_client, Client
from ..core.logging_config import get_logger
from ..core.exceptions import (
    StorageErrorException,
    SupabaseUnavailableException,
    FileTooLargeException,
    InvalidMimeTypeException
)
from ..core.file_validator import FileValidator

logger = get_logger("services.supabase_storage")


def retry_on_failure(max_attempts: int = 3, delay_seconds: float = 1.0, backoff_multiplier: float = 2.0):
    """
    Decorator to retry a function on transient failures.
    
    Implements exponential backoff for network-related errors.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay_seconds: Initial delay between retries in seconds (default: 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if it's a transient error worth retrying
                    is_transient = any(keyword in error_str for keyword in [
                        'timeout', 'connection', 'network', 'temporary',
                        'unavailable', 'service', '503', '502', '504'
                    ])
                    
                    if not is_transient or attempt >= max_attempts:
                        # Not a transient error or max attempts reached
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempt(s): {e}",
                            exc_info=True
                        )
                        raise
                    
                    # Log retry attempt
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )
                    
                    # Wait before retry with exponential backoff
                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class SupabaseStorageService:
    """
    Centralized Supabase storage service.
    ALL file uploads, downloads, and management MUST go through this service.
    """
    
    def __init__(self):
        """Initialize Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.client: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase storage service initialized")
        
        # Bucket names
        self.UPLOADS_BUCKET = "uploads"
        self.RESULTS_BUCKET = "results"
        self.GENERATED_BUCKET = "generated"
        self.WARDROBE_BUCKET = "wardrobe"
        self.USER_GARMENTS_BUCKET = "user-garments"
        self.USER_IMAGES_BUCKET = "user-images"
        
        # Initialize file validator
        self.file_validator = FileValidator(max_size_mb=10)
    
    @retry_on_failure(max_attempts=3, delay_seconds=1.0, backoff_multiplier=2.0)
    def upload_image(
        self,
        image: Image.Image,
        bucket: str,
        path: str,
        content_type: str = "image/png",
        max_size_mb: float = 5.0,
        quality: int = 85
    ) -> str:
        """
        Upload PIL Image to Supabase storage with automatic compression and retry logic.
        
        Retries up to 3 times on transient network failures with exponential backoff.
        
        Args:
            image: PIL Image object
            bucket: Bucket name
            path: File path in bucket
            content_type: MIME type
            max_size_mb: Maximum file size in MB (default 5MB)
            quality: JPEG quality for compression (1-100, default 85)
            
        Returns:
            Public URL of uploaded file
            
        Raises:
            RuntimeError: If upload fails after all retry attempts
        """
        try:
            # Compress image if needed
            compressed_image = self._compress_image(image, max_size_mb, quality)
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            
            # Use JPEG for better compression, PNG for transparency
            if compressed_image.mode in ('RGBA', 'LA', 'P'):
                # Has transparency, use PNG with optimization
                compressed_image.save(img_byte_arr, format='PNG', optimize=True)
                content_type = "image/png"
            else:
                # No transparency, use JPEG for better compression
                if compressed_image.mode != 'RGB':
                    compressed_image = compressed_image.convert('RGB')
                compressed_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                content_type = "image/jpeg"
                # Update path extension if needed
                if path.endswith('.png'):
                    path = path.rsplit('.', 1)[0] + '.jpg'
            
            img_byte_arr.seek(0)
            file_data = img_byte_arr.getvalue()
            file_size_mb = len(file_data) / (1024 * 1024)
            
            logger.info(f"Compressed image to {file_size_mb:.2f}MB for upload")
            
            # Upload to Supabase (will retry on transient failures)
            response = self.client.storage.from_(bucket).upload(
                path=path,
                file=file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(path)
            
            logger.info(f"Uploaded image to {bucket}/{path} ({file_size_mb:.2f}MB)")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image to Supabase: {e}", exc_info=True)
            raise RuntimeError(f"Supabase upload failed: {str(e)}")
    
    def _compress_image(
        self,
        image: Image.Image,
        max_size_mb: float = 5.0,
        quality: int = 85,
        max_dimension: int = 2048
    ) -> Image.Image:
        """
        Compress image to meet size requirements.
        
        Args:
            image: PIL Image object
            max_size_mb: Maximum file size in MB
            quality: JPEG quality (1-100)
            max_dimension: Maximum width or height
            
        Returns:
            Compressed PIL Image
        """
        # Resize if too large
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
        
        # Check size and reduce quality if needed
        for attempt_quality in range(quality, 50, -10):
            img_byte_arr = io.BytesIO()
            
            # Convert to RGB for JPEG compression test
            test_image = image.convert('RGB') if image.mode != 'RGB' else image
            test_image.save(img_byte_arr, format='JPEG', quality=attempt_quality, optimize=True)
            
            size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
            
            if size_mb <= max_size_mb:
                logger.info(f"Image compressed to {size_mb:.2f}MB at quality {attempt_quality}")
                return image
        
        # If still too large, resize more aggressively
        logger.warning(f"Image still too large, applying aggressive resize")
        width, height = image.size
        ratio = 0.7
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    @retry_on_failure(max_attempts=3, delay_seconds=1.0, backoff_multiplier=2.0)
    def upload_bytes(
        self,
        file_data: bytes,
        bucket: str,
        path: str,
        content_type: str = "image/png",
        compress: bool = True,
        max_size_mb: float = 5.0
    ) -> str:
        """
        Upload raw bytes to Supabase storage with optional compression and retry logic.
        
        Retries up to 3 times on transient network failures with exponential backoff.
        
        Args:
            file_data: File bytes
            bucket: Bucket name
            path: File path in bucket
            content_type: MIME type
            compress: Whether to compress image data
            max_size_mb: Maximum file size in MB if compressing
            
        Returns:
            Public URL of uploaded file
            
        Raises:
            RuntimeError: If upload fails after all retry attempts
        """
        try:
            # Compress if requested and it's an image
            if compress and content_type.startswith('image/'):
                try:
                    # Convert bytes to PIL Image
                    image = Image.open(io.BytesIO(file_data))
                    # Use upload_image for compression (which also has retry logic)
                    return self.upload_image(image, bucket, path, content_type, max_size_mb)
                except Exception as e:
                    logger.warning(f"Failed to compress image, uploading as-is: {e}")
            
            # Upload to Supabase (will retry on transient failures)
            response = self.client.storage.from_(bucket).upload(
                path=path,
                file=file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(path)
            
            file_size_mb = len(file_data) / (1024 * 1024)
            logger.info(f"Uploaded bytes to {bucket}/{path} ({file_size_mb:.2f}MB)")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload bytes to Supabase: {e}", exc_info=True)
            raise RuntimeError(f"Supabase upload failed: {str(e)}")
    
    def download_image(self, bucket: str, path: str) -> Image.Image:
        """
        Download image from Supabase storage.
        
        Args:
            bucket: Bucket name
            path: File path in bucket
            
        Returns:
            PIL Image object
        """
        try:
            # Download from Supabase
            file_data = self.client.storage.from_(bucket).download(path)
            
            if not file_data:
                raise FileNotFoundError(f"File not found: {bucket}/{path}")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(file_data))
            
            logger.info(f"Downloaded image from {bucket}/{path}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to download image from Supabase: {e}", exc_info=True)
            raise RuntimeError(f"Supabase download failed: {str(e)}")
    
    def download_bytes(self, bucket: str, path: str) -> bytes:
        """
        Download raw bytes from Supabase storage.
        
        Args:
            bucket: Bucket name
            path: File path in bucket
            
        Returns:
            File bytes
        """
        try:
            # Download from Supabase
            file_data = self.client.storage.from_(bucket).download(path)
            
            if not file_data:
                raise FileNotFoundError(f"File not found: {bucket}/{path}")
            
            logger.info(f"Downloaded bytes from {bucket}/{path}")
            return file_data
            
        except Exception as e:
            logger.error(f"Failed to download bytes from Supabase: {e}", exc_info=True)
            raise RuntimeError(f"Supabase download failed: {str(e)}")
    
    def delete_file(self, bucket: str, path: str) -> bool:
        """
        Delete file from Supabase storage.
        
        Args:
            bucket: Bucket name
            path: File path in bucket
            
        Returns:
            True if successful
        """
        try:
            response = self.client.storage.from_(bucket).remove([path])
            logger.info(f"Deleted file from {bucket}/{path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from Supabase: {e}", exc_info=True)
            return False
    
    def list_files(self, bucket: str, path: str = "") -> list:
        """
        List files in Supabase storage bucket.
        
        Args:
            bucket: Bucket name
            path: Optional path prefix
            
        Returns:
            List of file objects
        """
        try:
            response = self.client.storage.from_(bucket).list(path)
            logger.info(f"Listed files in {bucket}/{path}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to list files from Supabase: {e}", exc_info=True)
            return []
    
    def get_public_url(self, bucket: str, path: str) -> str:
        """
        Get public URL for a file.
        
        Args:
            bucket: Bucket name
            path: File path in bucket
            
        Returns:
            Public URL
        """
        return self.client.storage.from_(bucket).get_public_url(path)
    
    def _generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate unique filename to prevent collisions.
        
        Args:
            original_filename: Original filename with extension
            
        Returns:
            Unique filename with UUID prefix
        """
        # Extract extension
        name_parts = original_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            ext = ext.lower()
        else:
            name = original_filename
            ext = 'jpg'
        
        # Generate unique ID
        unique_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Create unique filename: {uuid}_{timestamp}.{ext}
        unique_filename = f"{unique_id}_{timestamp}.{ext}"
        
        return unique_filename
    
    async def upload_garment(
        self,
        user_id: str,
        file: UploadFile
    ) -> Dict[str, Any]:
        """
        Upload garment image to user-specific storage with retry logic.
        
        Validates file before upload and stores in user-garments/{userId}/garments/
        Supports JPEG, PNG, and WebP formats with clothing validation.
        Retries up to 3 times on transient network failures.
        
        Args:
            user_id: User ID for storage isolation
            file: FastAPI UploadFile object
            
        Returns:
            Dictionary with garment metadata including:
            - id: Unique garment ID
            - url: Public URL
            - name: Original filename
            - path: Storage path
            - uploaded_at: Upload timestamp
            - size_mb: File size in MB
            - content_type: MIME type
            
        Raises:
            FileTooLargeException: If file exceeds size limit
            InvalidMimeTypeException: If file type not allowed
            StorageErrorException: If upload fails after all retry attempts
        """
        try:
            # Validate file format (JPEG, PNG, WebP)
            validation_result = await self.file_validator.validate_file(file)
            
            if not validation_result.valid:
                raise StorageErrorException(
                    operation="upload_garment",
                    reason=validation_result.error_message or "Validation failed"
                )
            
            # Validate that the image contains clothing
            # Read file content for validation
            file_content = await file.read()
            await file.seek(0)  # Reset for potential reuse
            
            # Validate clothing content using AI (optional, can be added later)
            # For now, we rely on user upload and file format validation
            is_valid_clothing = await self._validate_clothing_content(file_content, file.content_type)
            
            if not is_valid_clothing:
                logger.warning(f"Clothing validation failed for user {user_id}")
                raise StorageErrorException(
                    operation="upload_garment",
                    reason="Image does not appear to contain clothing items"
                )
            
            # Generate unique filename
            unique_filename = self._generate_unique_filename(
                validation_result.sanitized_filename or file.filename or "garment.jpg"
            )
            
            # Construct storage path: user-garments/{userId}/garments/{unique_filename}
            storage_path = f"{user_id}/garments/{unique_filename}"
            
            # Upload to Supabase with retry logic
            max_attempts = 3
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.client.storage.from_(self.USER_GARMENTS_BUCKET).upload(
                        path=storage_path,
                        file=file_content,
                        file_options={
                            "content-type": file.content_type,
                            "upsert": False  # Don't overwrite existing files
                        }
                    )
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if it's a transient error worth retrying
                    is_transient = any(keyword in error_str for keyword in [
                        'timeout', 'connection', 'network', 'temporary',
                        'unavailable', 'service', '503', '502', '504'
                    ])
                    
                    if not is_transient or attempt >= max_attempts:
                        # Not a transient error or max attempts reached
                        logger.error(
                            f"Garment upload failed after {attempt} attempt(s): {e}",
                            exc_info=True
                        )
                        raise StorageErrorException(
                            operation="upload_garment",
                            reason=f"Upload failed after {attempt} attempts: {str(e)}"
                        )
                    
                    # Log retry attempt
                    logger.warning(
                        f"Garment upload failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {attempt}s: {e}"
                    )
                    
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(attempt)
            
            # Get public URL
            public_url = self.client.storage.from_(self.USER_GARMENTS_BUCKET).get_public_url(storage_path)
            
            # Extract garment ID from unique filename (UUID part)
            garment_id = unique_filename.split('_')[0]
            
            metadata = {
                "id": garment_id,
                "url": public_url,
                "name": validation_result.sanitized_filename or file.filename,
                "path": storage_path,
                "uploaded_at": datetime.utcnow().isoformat(),
                "size_mb": validation_result.file_size_mb,
                "content_type": file.content_type
            }
            
            logger.info(f"Uploaded garment for user {user_id}: {garment_id}")
            return metadata
            
        except (FileTooLargeException, InvalidMimeTypeException):
            raise
        except Exception as e:
            logger.error(f"Failed to upload garment for user {user_id}: {e}", exc_info=True)
            raise StorageErrorException(
                operation="upload_garment",
                reason=str(e)
            )
    
    async def _validate_clothing_content(self, file_content: bytes, content_type: str) -> bool:
        """
        Validate that the image contains clothing items.
        
        This is a basic validation that checks if the image can be opened.
        More sophisticated validation using AI models can be added later.
        
        Args:
            file_content: Image file bytes
            content_type: MIME type of the image
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Basic validation: ensure image can be opened
            image = Image.open(io.BytesIO(file_content))
            
            # Check image dimensions (clothing images should be reasonable size)
            width, height = image.size
            if width < 50 or height < 50:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            
            if width > 4096 or height > 4096:
                logger.warning(f"Image too large: {width}x{height}")
                return False
            
            # TODO: Add AI-based clothing detection using Gemini Vision API
            # For now, we accept all valid images
            return True
            
        except Exception as e:
            logger.error(f"Clothing validation error: {e}")
            return False
    
    def list_garments(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all garments for a specific user with strict user isolation.
        
        Ensures that only garments belonging to the specified user are returned.
        
        Args:
            user_id: User ID for storage isolation
            
        Returns:
            List of garment metadata dictionaries
            
        Raises:
            SupabaseUnavailableException: If Supabase is unavailable
            StorageErrorException: If listing fails
        """
        try:
            # List files in user's garment directory (user isolation enforced by path)
            garments_path = f"{user_id}/garments"
            
            files = self.client.storage.from_(self.USER_GARMENTS_BUCKET).list(garments_path)
            
            garments = []
            for file_obj in files:
                # Skip directories
                if file_obj.get('id') is None:
                    continue
                
                file_name = file_obj.get('name', '')
                file_path = f"{garments_path}/{file_name}"
                
                # Extract garment ID from filename (UUID part)
                garment_id = file_name.split('_')[0] if '_' in file_name else file_name.split('.')[0]
                
                # Get public URL
                public_url = self.client.storage.from_(self.USER_GARMENTS_BUCKET).get_public_url(file_path)
                
                metadata = {
                    "id": garment_id,
                    "url": public_url,
                    "name": file_name,
                    "path": file_path,
                    "uploaded_at": file_obj.get('created_at', ''),
                    "size_bytes": file_obj.get('metadata', {}).get('size', 0)
                }
                
                garments.append(metadata)
            
            logger.info(f"Listed {len(garments)} garments for user {user_id} (user isolation enforced)")
            return garments
            
        except Exception as e:
            logger.error(f"Failed to list garments for user {user_id}: {e}", exc_info=True)
            
            # Check if it's a connection error
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise SupabaseUnavailableException(
                    details={"user_id": user_id, "error": str(e)}
                )
            
            raise StorageErrorException(
                operation="list_garments",
                reason=str(e),
                details={"user_id": user_id}
            )
    
    def delete_garment(self, user_id: str, garment_id: str) -> bool:
        """
        Delete a specific garment for a user.
        
        Finds the garment by ID and removes it from storage.
        
        Args:
            user_id: User ID for storage isolation
            garment_id: Garment ID (UUID from filename)
            
        Returns:
            True if deletion successful, False otherwise
            
        Raises:
            StorageErrorException: If deletion fails
        """
        try:
            # List garments to find the one with matching ID
            garments = self.list_garments(user_id)
            
            # Find garment with matching ID
            garment_to_delete = None
            for garment in garments:
                if garment['id'] == garment_id:
                    garment_to_delete = garment
                    break
            
            if not garment_to_delete:
                logger.warning(f"Garment {garment_id} not found for user {user_id}")
                return False
            
            # Delete from Supabase
            file_path = garment_to_delete['path']
            response = self.client.storage.from_(self.USER_GARMENTS_BUCKET).remove([file_path])
            
            logger.info(f"Deleted garment {garment_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete garment {garment_id} for user {user_id}: {e}", exc_info=True)
            raise StorageErrorException(
                operation="delete_garment",
                reason=str(e),
                details={"user_id": user_id, "garment_id": garment_id}
            )
    
    async def upload_personal_image(
        self,
        user_id: str,
        file: UploadFile,
        image_type: str = "profile"
    ) -> Dict[str, Any]:
        """
        Upload user's personal image (profile photo) with retry logic.
        
        Validates file before upload and stores in user-images/{userId}/personal/
        Retries up to 3 times on transient network failures.
        
        Args:
            user_id: User ID for storage isolation
            file: FastAPI UploadFile object
            image_type: Type of image (profile, full-body, etc.)
            
        Returns:
            Dictionary with image metadata including:
            - url: Public URL
            - path: Storage path
            - uploaded_at: Upload timestamp
            
        Raises:
            FileTooLargeException: If file exceeds size limit
            InvalidMimeTypeException: If file type not allowed
            StorageErrorException: If upload fails after all retry attempts
        """
        try:
            # Validate file
            validation_result = await self.file_validator.validate_file(file)
            
            if not validation_result.valid:
                raise StorageErrorException(
                    operation="upload_personal_image",
                    reason=validation_result.error_message or "Validation failed"
                )
            
            # Construct storage path: user-images/{userId}/personal/{image_type}.{ext}
            ext = validation_result.sanitized_filename.rsplit('.', 1)[-1] if '.' in validation_result.sanitized_filename else 'jpg'
            storage_path = f"{user_id}/personal/{image_type}.{ext}"
            
            # Read file content
            file_content = await file.read()
            await file.seek(0)
            
            # Upload to Supabase with retry logic
            max_attempts = 3
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.client.storage.from_(self.USER_IMAGES_BUCKET).upload(
                        path=storage_path,
                        file=file_content,
                        file_options={
                            "content-type": file.content_type,
                            "upsert": "true"  # Allow overwriting existing personal image
                        }
                    )
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if it's a transient error worth retrying
                    is_transient = any(keyword in error_str for keyword in [
                        'timeout', 'connection', 'network', 'temporary',
                        'unavailable', 'service', '503', '502', '504'
                    ])
                    
                    if not is_transient or attempt >= max_attempts:
                        # Not a transient error or max attempts reached
                        logger.error(
                            f"Personal image upload failed after {attempt} attempt(s): {e}",
                            exc_info=True
                        )
                        raise StorageErrorException(
                            operation="upload_personal_image",
                            reason=f"Upload failed after {attempt} attempts: {str(e)}"
                        )
                    
                    # Log retry attempt
                    logger.warning(
                        f"Personal image upload failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {attempt}s: {e}"
                    )
                    
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(attempt)
            
            # Get public URL
            public_url = self.client.storage.from_(self.USER_IMAGES_BUCKET).get_public_url(storage_path)
            
            metadata = {
                "url": public_url,
                "path": storage_path,
                "uploaded_at": datetime.utcnow().isoformat(),
                "size_mb": validation_result.file_size_mb,
                "content_type": file.content_type,
                "image_type": image_type
            }
            
            logger.info(f"Uploaded personal image for user {user_id}: {image_type}")
            return metadata
            
        except (FileTooLargeException, InvalidMimeTypeException):
            raise
        except Exception as e:
            logger.error(f"Failed to upload personal image for user {user_id}: {e}", exc_info=True)
            raise StorageErrorException(
                operation="upload_personal_image",
                reason=str(e)
            )
    
    def get_personal_image(self, user_id: str, image_type: str = "profile") -> Optional[str]:
        """
        Get user's personal image URL.
        
        Args:
            user_id: User ID for storage isolation
            image_type: Type of image (profile, full-body, etc.)
            
        Returns:
            Public URL if image exists, None otherwise
        """
        try:
            # Try common image extensions
            for ext in ['jpg', 'jpeg', 'png', 'webp']:
                storage_path = f"{user_id}/personal/{image_type}.{ext}"
                
                # Check if file exists by trying to list it
                try:
                    files = self.client.storage.from_(self.USER_IMAGES_BUCKET).list(f"{user_id}/personal")
                    
                    # Check if our file exists in the list
                    file_name = f"{image_type}.{ext}"
                    if any(f.get('name') == file_name for f in files):
                        public_url = self.client.storage.from_(self.USER_IMAGES_BUCKET).get_public_url(storage_path)
                        logger.info(f"Found personal image for user {user_id}: {image_type}")
                        return public_url
                except:
                    continue
            
            logger.info(f"No personal image found for user {user_id}: {image_type}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get personal image for user {user_id}: {e}", exc_info=True)
            return None
    
    # ========== DATABASE OPERATIONS ==========
    
    def list_user_garments_db(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all garments for a user from Supabase database table.
        Enforces strict user isolation by filtering on user_id.
        
        Args:
            user_id: User ID
            
        Returns:
            List of garment records from database (only for specified user)
        """
        try:
            logger.info(f"Fetching garments from database for user: {user_id}")
            
            # Query garments table with user_id filter (user isolation)
            response = self.client.table('garments').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
            
            garments = response.data if response.data else []
            logger.info(f"Found {len(garments)} garments in database for user {user_id} (user isolation enforced)")
            
            return garments
            
        except Exception as e:
            logger.error(f"Failed to list user garments from database: {e}", exc_info=True)
            # Return empty list instead of raising to allow graceful degradation
            return []
    
    def list_user_results_db(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all try-on results for a user from Supabase database table.
        Enforces strict user isolation by filtering on user_id.
        
        Args:
            user_id: User ID
            
        Returns:
            List of try-on result records from database (only for specified user)
        """
        try:
            logger.info(f"Fetching try-on results from database for user: {user_id}")
            
            # Query tryon_results table with user_id filter (user isolation)
            response = self.client.table('tryon_results').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
            
            results = response.data if response.data else []
            logger.info(f"Found {len(results)} try-on results in database for user {user_id} (user isolation enforced)")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to list user results from database: {e}", exc_info=True)
            # Return empty list instead of raising to allow graceful degradation
            return []
    
    def save_garment_record_db(
        self, 
        user_id: str, 
        url: str, 
        name: str, 
        metadata: Optional[Dict] = None,
        file_size_bytes: Optional[int] = None,
        content_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Save garment record to Supabase database table with metadata.
        
        Stores timestamps, file sizes, and content type for analytics and tracking.
        
        Args:
            user_id: User ID
            url: Public URL to garment image
            name: Garment name
            metadata: Optional metadata (JSONB)
            file_size_bytes: File size in bytes
            content_type: MIME type (e.g., image/jpeg, image/png, image/webp)
            
        Returns:
            Created garment record or None if failed
        """
        try:
            logger.info(f"Saving garment record to database for user: {user_id}")
            
            # Prepare metadata with timestamp
            enhanced_metadata = metadata or {}
            enhanced_metadata['uploaded_at'] = datetime.utcnow().isoformat()
            
            garment_data = {
                'user_id': user_id,
                'url': url,
                'name': name,
                'metadata': enhanced_metadata,
                'file_size_bytes': file_size_bytes or 0,
                'content_type': content_type or 'image/jpeg',
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = self.client.table('garments').insert(garment_data).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Garment record saved to database: {response.data[0].get('id')} (size: {file_size_bytes} bytes)")
                return response.data[0]
            else:
                logger.warning("Failed to save garment record to database")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save garment record to database: {e}", exc_info=True)
            return None
    
    def save_tryon_result_db(
        self, 
        user_id: str, 
        personal_image_url: str, 
        garment_url: str, 
        result_url: str, 
        metadata: Optional[Dict] = None,
        model_version: str = "catvton-v1",
        processing_time_seconds: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Save try-on result to Supabase database table with metadata.
        
        Stores timestamps, model version, processing time, and other metadata
        for analytics and tracking.
        
        Args:
            user_id: User ID
            personal_image_url: URL to personal image
            garment_url: URL to garment image
            result_url: URL to result image
            metadata: Optional metadata (JSONB) - includes inference params
            model_version: Model version used (e.g., catvton-v1)
            processing_time_seconds: Processing time in seconds
            
        Returns:
            Created result record or None if failed
        """
        try:
            logger.info(f"Saving try-on result to database for user: {user_id}")
            
            # Prepare enhanced metadata with timestamp and processing info
            enhanced_metadata = metadata or {}
            enhanced_metadata['saved_at'] = datetime.utcnow().isoformat()
            enhanced_metadata['model_version'] = model_version
            enhanced_metadata['processing_time_seconds'] = processing_time_seconds
            
            result_data = {
                'user_id': user_id,
                'personal_image_url': personal_image_url,
                'garment_url': garment_url,
                'result_url': result_url,
                'status': 'completed',
                'metadata': enhanced_metadata,
                'model_version': model_version,
                'processing_time_seconds': processing_time_seconds,
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = self.client.table('tryon_results').insert(result_data).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(
                    f"Try-on result saved to database: {response.data[0].get('id')} "
                    f"(model: {model_version}, time: {processing_time_seconds:.2f}s)"
                )
                return response.data[0]
            else:
                logger.warning("Failed to save try-on result to database")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save try-on result to database: {e}", exc_info=True)
            return None


# Singleton instance
supabase_storage = SupabaseStorageService()
