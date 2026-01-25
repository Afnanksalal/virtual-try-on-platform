"""
Custom exception classes for the virtual try-on platform.

This module defines exception classes for different error categories,
each mapped to specific error codes for consistent error handling.
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(str, Enum):
    """Hierarchical error code system for consistent error handling."""
    
    # Authentication Errors (1xxx)
    AUTH_INVALID_TOKEN = "AUTH_1001"
    AUTH_EXPIRED_TOKEN = "AUTH_1002"
    AUTH_MISSING_TOKEN = "AUTH_1003"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_1004"
    
    # Validation Errors (2xxx)
    VALIDATION_INVALID_INPUT = "VAL_2001"
    VALIDATION_FILE_TOO_LARGE = "VAL_2002"
    VALIDATION_INVALID_MIME_TYPE = "VAL_2003"
    VALIDATION_CONTENT_MISMATCH = "VAL_2004"
    VALIDATION_INVALID_FILENAME = "VAL_2005"
    
    # Rate Limiting Errors (3xxx)
    RATE_LIMIT_EXCEEDED = "RATE_3001"
    
    # ML Service Errors (4xxx)
    ML_MODEL_NOT_LOADED = "ML_4001"
    ML_INFERENCE_FAILED = "ML_4002"
    ML_GPU_OUT_OF_MEMORY = "ML_4003"
    ML_INVALID_INPUT_DIMENSIONS = "ML_4004"
    
    # External Service Errors (5xxx)
    EXT_GEMINI_UNAVAILABLE = "EXT_5001"
    EXT_EBAY_UNAVAILABLE = "EXT_5002"
    EXT_SUPABASE_UNAVAILABLE = "EXT_5003"
    EXT_SERVICE_TIMEOUT = "EXT_5004"
    
    # System Errors (9xxx)
    SYS_INTERNAL_ERROR = "SYS_9001"
    SYS_CONFIGURATION_ERROR = "SYS_9002"
    SYS_STORAGE_ERROR = "SYS_9003"


class AppException(Exception):
    """Base exception class for all application exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


# Authentication Exceptions

class AuthenticationException(AppException):
    """Base class for authentication-related exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_code, message, details, status_code=401)


class InvalidTokenException(AuthenticationException):
    """Raised when an invalid authentication token is provided."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.AUTH_INVALID_TOKEN,
            "Invalid authentication token",
            details
        )


class ExpiredTokenException(AuthenticationException):
    """Raised when an expired authentication token is provided."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.AUTH_EXPIRED_TOKEN,
            "Authentication token has expired",
            details
        )


class MissingTokenException(AuthenticationException):
    """Raised when no authentication token is provided."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.AUTH_MISSING_TOKEN,
            "Authentication token is required",
            details
        )


class InsufficientPermissionsException(AppException):
    """Raised when user lacks required permissions."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            "Insufficient permissions to access this resource",
            details,
            status_code=403
        )


# Validation Exceptions

class ValidationException(AppException):
    """Base class for validation-related exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_code, message, details, status_code=422)


class InvalidInputException(ValidationException):
    """Raised when input validation fails."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.VALIDATION_INVALID_INPUT,
            "Invalid input provided",
            details
        )


class FileTooLargeException(ValidationException):
    """Raised when uploaded file exceeds size limit."""
    
    def __init__(self, file_size_mb: float, max_size_mb: float, field: str = "file"):
        super().__init__(
            ErrorCode.VALIDATION_FILE_TOO_LARGE,
            f"File size exceeds maximum allowed size of {max_size_mb}MB",
            {
                "file_size_mb": file_size_mb,
                "max_size_mb": max_size_mb,
                "field": field
            }
        )
        self.status_code = 413


class InvalidMimeTypeException(ValidationException):
    """Raised when file MIME type is not allowed."""
    
    def __init__(self, mime_type: str, allowed_types: list, field: str = "file"):
        super().__init__(
            ErrorCode.VALIDATION_INVALID_MIME_TYPE,
            f"File type '{mime_type}' is not allowed",
            {
                "mime_type": mime_type,
                "allowed_types": allowed_types,
                "field": field
            }
        )
        self.status_code = 415


class ContentMismatchException(ValidationException):
    """Raised when file content doesn't match declared MIME type."""
    
    def __init__(self, declared_type: str, actual_type: str, field: str = "file"):
        super().__init__(
            ErrorCode.VALIDATION_CONTENT_MISMATCH,
            "File content does not match declared type",
            {
                "declared_type": declared_type,
                "actual_type": actual_type,
                "field": field
            }
        )
        self.status_code = 400


class InvalidFilenameException(ValidationException):
    """Raised when filename contains invalid characters."""
    
    def __init__(self, filename: str, field: str = "file"):
        super().__init__(
            ErrorCode.VALIDATION_INVALID_FILENAME,
            "Filename contains invalid characters",
            {
                "filename": filename,
                "field": field
            }
        )
        self.status_code = 400


# Rate Limiting Exceptions

class RateLimitExceededException(AppException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int, details: Optional[Dict[str, Any]] = None):
        merged_details = details or {}
        merged_details["retry_after"] = retry_after
        super().__init__(
            ErrorCode.RATE_LIMIT_EXCEEDED,
            f"Rate limit exceeded. Please retry after {retry_after} seconds",
            merged_details,
            status_code=429
        )


# ML Service Exceptions

class MLServiceException(AppException):
    """Base class for ML service-related exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_code, message, details, status_code=500)


class ModelNotLoadedException(MLServiceException):
    """Raised when a required ML model is not loaded."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        merged_details = details or {}
        merged_details["model_name"] = model_name
        super().__init__(
            ErrorCode.ML_MODEL_NOT_LOADED,
            f"Model '{model_name}' is not loaded",
            merged_details
        )


class InferenceFailedException(MLServiceException):
    """Raised when ML inference fails."""
    
    def __init__(self, model_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        merged_details = details or {}
        merged_details.update({
            "model_name": model_name,
            "reason": reason
        })
        super().__init__(
            ErrorCode.ML_INFERENCE_FAILED,
            f"Inference failed for model '{model_name}': {reason}",
            merged_details
        )


class GPUOutOfMemoryException(MLServiceException):
    """Raised when GPU runs out of memory."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.ML_GPU_OUT_OF_MEMORY,
            "GPU out of memory. Please try again later or reduce batch size",
            details
        )


class InvalidInputDimensionsException(MLServiceException):
    """Raised when input dimensions are invalid for the model."""
    
    def __init__(
        self,
        expected_dimensions: str,
        actual_dimensions: str,
        details: Optional[Dict[str, Any]] = None
    ):
        merged_details = details or {}
        merged_details.update({
            "expected_dimensions": expected_dimensions,
            "actual_dimensions": actual_dimensions
        })
        super().__init__(
            ErrorCode.ML_INVALID_INPUT_DIMENSIONS,
            f"Invalid input dimensions. Expected {expected_dimensions}, got {actual_dimensions}",
            merged_details
        )


# External Service Exceptions

class ExternalServiceException(AppException):
    """Base class for external service-related exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_code, message, details, status_code=503)


class GeminiUnavailableException(ExternalServiceException):
    """Raised when Gemini API is unavailable."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.EXT_GEMINI_UNAVAILABLE,
            "Gemini Vision API is currently unavailable",
            details
        )


class EbayUnavailableException(ExternalServiceException):
    """Raised when eBay API is unavailable."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.EXT_EBAY_UNAVAILABLE,
            "eBay API is currently unavailable",
            details
        )


class SupabaseUnavailableException(ExternalServiceException):
    """Raised when Supabase is unavailable."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.EXT_SUPABASE_UNAVAILABLE,
            "Supabase service is currently unavailable",
            details
        )


class ServiceTimeoutException(ExternalServiceException):
    """Raised when external service call times out."""
    
    def __init__(self, service_name: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None):
        merged_details = details or {}
        merged_details.update({
            "service_name": service_name,
            "timeout_seconds": timeout_seconds
        })
        super().__init__(
            ErrorCode.EXT_SERVICE_TIMEOUT,
            f"Request to {service_name} timed out after {timeout_seconds} seconds",
            merged_details
        )


# System Exceptions

class SystemException(AppException):
    """Base class for system-level exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_code, message, details, status_code=500)


class InternalErrorException(SystemException):
    """Raised for unexpected internal errors."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.SYS_INTERNAL_ERROR,
            "An internal error occurred. Please try again later",
            details
        )


class ConfigurationErrorException(SystemException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, reason: str, details: Optional[Dict[str, Any]] = None):
        merged_details = details or {}
        merged_details.update({
            "config_key": config_key,
            "reason": reason
        })
        super().__init__(
            ErrorCode.SYS_CONFIGURATION_ERROR,
            f"Configuration error for '{config_key}': {reason}",
            merged_details
        )


class StorageErrorException(SystemException):
    """Raised when file storage operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        merged_details = details or {}
        merged_details.update({
            "operation": operation,
            "reason": reason
        })
        super().__init__(
            ErrorCode.SYS_STORAGE_ERROR,
            f"Storage operation '{operation}' failed: {reason}",
            merged_details
        )
