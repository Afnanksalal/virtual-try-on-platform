"""
Supabase Storage Service - ALL file operations go through Supabase.
NO local file storage allowed.
"""

import os
import io
from typing import Optional, BinaryIO
from PIL import Image
from supabase import create_client, Client
from ..core.logging_config import get_logger

logger = get_logger("services.supabase_storage")


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
    
    def upload_image(
        self,
        image: Image.Image,
        bucket: str,
        path: str,
        content_type: str = "image/png"
    ) -> str:
        """
        Upload PIL Image to Supabase storage.
        
        Args:
            image: PIL Image object
            bucket: Bucket name
            path: File path in bucket
            content_type: MIME type
            
        Returns:
            Public URL of uploaded file
        """
        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            file_data = img_byte_arr.getvalue()
            
            # Upload to Supabase
            response = self.client.storage.from_(bucket).upload(
                path=path,
                file=file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": True
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(path)
            
            logger.info(f"Uploaded image to {bucket}/{path}")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image to Supabase: {e}", exc_info=True)
            raise RuntimeError(f"Supabase upload failed: {str(e)}")
    
    def upload_bytes(
        self,
        file_data: bytes,
        bucket: str,
        path: str,
        content_type: str = "image/png"
    ) -> str:
        """
        Upload raw bytes to Supabase storage.
        
        Args:
            file_data: File bytes
            bucket: Bucket name
            path: File path in bucket
            content_type: MIME type
            
        Returns:
            Public URL of uploaded file
        """
        try:
            # Upload to Supabase
            response = self.client.storage.from_(bucket).upload(
                path=path,
                file=file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": True
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(path)
            
            logger.info(f"Uploaded bytes to {bucket}/{path}")
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


# Singleton instance
supabase_storage = SupabaseStorageService()
