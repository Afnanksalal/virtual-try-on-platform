"""
Temporary File Manager Service

Manages temporary file storage with secure token-based access and automatic expiration.
Used primarily for 3D model downloads (GLB/OBJ/PLY files).
"""

import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class TempFileManager:
    """
    Manages temporary file storage with token-based access and expiration.
    
    Features:
    - Cryptographically secure UUID tokens
    - Configurable expiration time (default 1 hour)
    - Thread-safe operations
    - Automatic metadata tracking (creation time, expiration, original filename)
    - Files stored in data/temp/ directory
    """
    
    def __init__(self, temp_dir: str = "data/temp", default_expiry_seconds: int = 3600):
        """
        Initialize the TempFileManager.
        
        Args:
            temp_dir: Directory to store temporary files (default: data/temp)
            default_expiry_seconds: Default expiration time in seconds (default: 3600 = 1 hour)
        """
        self.temp_dir = Path(temp_dir)
        self.default_expiry_seconds = default_expiry_seconds
        self.metadata_file = self.temp_dir / "metadata.json"
        self._lock = threading.Lock()
        
        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self._metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
        
        logger.info(f"TempFileManager initialized with temp_dir={temp_dir}, default_expiry={default_expiry_seconds}s")
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def store_temp_file(
        self,
        file_bytes: bytes,
        original_filename: str,
        expires_in_seconds: Optional[int] = None
    ) -> str:
        """
        Store a temporary file and return a secure download token.
        
        Args:
            file_bytes: The file content as bytes
            original_filename: Original filename (for metadata and content-type detection)
            expires_in_seconds: Expiration time in seconds (uses default if None)
        
        Returns:
            str: Secure UUID token for downloading the file
        
        Raises:
            ValueError: If file_bytes is empty
            IOError: If file write fails
        """
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty")
        
        # Use default expiry if not specified
        expiry_seconds = expires_in_seconds if expires_in_seconds is not None else self.default_expiry_seconds
        
        # Generate secure UUID token
        token = str(uuid.uuid4())
        
        # Calculate expiration time
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=expiry_seconds)
        
        # Create file path using token as filename
        file_extension = Path(original_filename).suffix
        file_path = self.temp_dir / f"{token}{file_extension}"
        
        try:
            # Thread-safe file write and metadata update
            with self._lock:
                # Write file
                with open(file_path, 'wb') as f:
                    f.write(file_bytes)
                
                # Store metadata
                self._metadata[token] = {
                    "original_filename": original_filename,
                    "file_path": str(file_path),
                    "created_at": created_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "file_size": len(file_bytes),
                    "extension": file_extension
                }
                
                # Save metadata to disk
                self._save_metadata()
            
            logger.info(f"Stored temp file: token={token}, filename={original_filename}, size={len(file_bytes)} bytes, expires_at={expires_at}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to store temp file: {e}")
            # Clean up partial file if it exists
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            raise IOError(f"Failed to store temporary file: {e}")
    
    def get_temp_file(self, token: str) -> Optional[Tuple[bytes, str]]:
        """
        Retrieve a temporary file by its token.
        
        Args:
            token: The UUID token returned by store_temp_file
        
        Returns:
            Optional[Tuple[bytes, str]]: Tuple of (file_bytes, original_filename) if found and not expired,
                                         None if token is invalid or file has expired
        """
        with self._lock:
            # Check if token exists in metadata
            if token not in self._metadata:
                logger.warning(f"Token not found: {token}")
                return None
            
            metadata = self._metadata[token]
            
            # Check if file has expired
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.utcnow() > expires_at:
                logger.info(f"Token expired: {token}, expired_at={expires_at}")
                # Clean up expired file
                self._delete_file(token)
                return None
            
            # Read file
            file_path = Path(metadata["file_path"])
            if not file_path.exists():
                logger.error(f"File not found for token: {token}, path={file_path}")
                # Clean up metadata for missing file
                del self._metadata[token]
                self._save_metadata()
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                original_filename = metadata["original_filename"]
                logger.info(f"Retrieved temp file: token={token}, filename={original_filename}, size={len(file_bytes)} bytes")
                return (file_bytes, original_filename)
                
            except Exception as e:
                logger.error(f"Failed to read temp file: token={token}, error={e}")
                return None
    
    def _delete_file(self, token: str) -> bool:
        """
        Delete a temporary file and its metadata (internal method).
        
        Args:
            token: The UUID token
        
        Returns:
            bool: True if file was deleted, False otherwise
        """
        if token not in self._metadata:
            return False
        
        metadata = self._metadata[token]
        file_path = Path(metadata["file_path"])
        
        # Delete file if it exists
        if file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Deleted temp file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file: {file_path}, error={e}")
        
        # Remove metadata
        del self._metadata[token]
        self._save_metadata()
        
        return True
    
    def cleanup_expired_files(self) -> int:
        """
        Clean up all expired temporary files.
        
        Returns:
            int: Number of files deleted
        """
        with self._lock:
            current_time = datetime.utcnow()
            expired_tokens = []
            
            # Find expired tokens
            for token, metadata in self._metadata.items():
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                if current_time > expires_at:
                    expired_tokens.append(token)
            
            # Delete expired files
            deleted_count = 0
            for token in expired_tokens:
                if self._delete_file(token):
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired temp files")
            
            return deleted_count
    
    def get_file_metadata(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a temporary file without reading the file content.
        
        Args:
            token: The UUID token
        
        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary if token exists, None otherwise
        """
        with self._lock:
            if token in self._metadata:
                return self._metadata[token].copy()
            return None
    
    def get_all_files_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all temporary files (for debugging/monitoring).
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping tokens to their metadata
        """
        with self._lock:
            return {token: metadata.copy() for token, metadata in self._metadata.items()}
