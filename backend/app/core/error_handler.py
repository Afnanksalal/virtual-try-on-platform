"""
Centralized error handling and response formatting.

This module provides utilities for handling exceptions and formatting
error responses consistently across the application.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Dict, Any, Optional
import traceback

from app.core.exceptions import AppException, ErrorCode
from app.core.logging_config import get_context_logger


def classify_error_type(exception: Exception) -> str:
    """
    Classify an error as user_error or system_error.
    
    Args:
        exception: The exception to classify
        
    Returns:
        "user_error" or "system_error"
    """
    # User errors (4xx status codes)
    user_error_types = (
        RequestValidationError,
    )
    
    if isinstance(exception, AppException):
        # Use status code to determine error type
        if 400 <= exception.status_code < 500:
            return "user_error"
        else:
            return "system_error"
    
    if isinstance(exception, user_error_types):
        return "user_error"
    
    if isinstance(exception, StarletteHTTPException):
        if 400 <= exception.status_code < 500:
            return "user_error"
        else:
            return "system_error"
    
    # Default to system error for unexpected exceptions
    return "system_error"


def format_error_response(
    error_code: str,
    message: str,
    error_type: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format error response in standardized structure.
    
    Args:
        error_code: Error code constant
        message: User-friendly error message
        error_type: "user_error" or "system_error"
        details: Additional error details
        request_id: Request ID for tracking
        
    Returns:
        Formatted error response dictionary
    """
    response = {
        "error": {
            "type": error_type,
            "code": error_code,
            "message": message,
            "details": details or {},
        }
    }
    
    if request_id:
        response["error"]["request_id"] = request_id
    
    return response


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """
    Handle AppException instances with structured error responses.
    
    Args:
        request: FastAPI request object
        exc: AppException instance
        
    Returns:
        JSONResponse with formatted error
    """
    request_id = getattr(request.state, "request_id", None)
    logger = get_context_logger("error_handler", request_id)
    
    # Determine error type
    error_type = classify_error_type(exc)
    
    # Log the error with context
    log_extra = {
        "error_code": exc.error_code.value,
        "error_type": error_type,
        "status_code": exc.status_code,
        "path": request.url.path,
        "method": request.method,
    }
    
    if error_type == "system_error":
        # Log system errors with full stack trace
        logger.error(
            f"System error occurred: {exc.message}",
            extra=log_extra,
            exc_info=True
        )
    else:
        # Log user errors at warning level without stack trace
        logger.warning(
            f"User error: {exc.message}",
            extra=log_extra
        )
    
    # Format response
    response_data = format_error_response(
        error_code=exc.error_code.value,
        message=exc.message,
        error_type=error_type,
        details=exc.details,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors with structured responses.
    
    Args:
        request: FastAPI request object
        exc: RequestValidationError instance
        
    Returns:
        JSONResponse with formatted validation error
    """
    request_id = getattr(request.state, "request_id", None)
    logger = get_context_logger("error_handler", request_id)
    
    # Extract validation errors
    errors = exc.errors()
    
    # Format validation details
    validation_details = {
        "validation_errors": [
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            }
            for error in errors
        ]
    }
    
    # Log validation error
    logger.warning(
        f"Validation error on {request.method} {request.url.path}",
        extra={
            "error_type": "user_error",
            "validation_errors": validation_details["validation_errors"],
            "path": request.url.path,
            "method": request.method,
        }
    )
    
    # Format response
    response_data = format_error_response(
        error_code=ErrorCode.VALIDATION_INVALID_INPUT.value,
        message="Request validation failed",
        error_type="user_error",
        details=validation_details,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handle HTTP exceptions with structured responses.
    
    Args:
        request: FastAPI request object
        exc: HTTPException instance
        
    Returns:
        JSONResponse with formatted error
    """
    request_id = getattr(request.state, "request_id", None)
    logger = get_context_logger("error_handler", request_id)
    
    # Determine error type
    error_type = classify_error_type(exc)
    
    # Map status code to error code
    error_code_map = {
        400: ErrorCode.VALIDATION_INVALID_INPUT,
        401: ErrorCode.AUTH_INVALID_TOKEN,
        403: ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
        404: "NOT_FOUND",
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.SYS_INTERNAL_ERROR,
        503: ErrorCode.EXT_SERVICE_TIMEOUT,
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCode.SYS_INTERNAL_ERROR)
    if isinstance(error_code, ErrorCode):
        error_code = error_code.value
    
    # Log the error
    log_extra = {
        "error_code": error_code,
        "error_type": error_type,
        "status_code": exc.status_code,
        "path": request.url.path,
        "method": request.method,
    }
    
    if error_type == "system_error":
        logger.error(
            f"HTTP error {exc.status_code}: {exc.detail}",
            extra=log_extra
        )
    else:
        logger.warning(
            f"HTTP error {exc.status_code}: {exc.detail}",
            extra=log_extra
        )
    
    # Format response
    response_data = format_error_response(
        error_code=error_code,
        message=str(exc.detail),
        error_type=error_type,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions with structured responses.
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSONResponse with formatted error
    """
    request_id = getattr(request.state, "request_id", None)
    logger = get_context_logger("error_handler", request_id)
    
    # Log the unexpected error with full stack trace
    logger.critical(
        f"Unexpected error: {str(exc)}",
        extra={
            "error_type": "system_error",
            "exception_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
            "stack_trace": traceback.format_exc(),
        },
        exc_info=True
    )
    
    # Format response (hide internal details from user)
    response_data = format_error_response(
        error_code=ErrorCode.SYS_INTERNAL_ERROR.value,
        message="An internal error occurred. Please try again later",
        error_type="system_error",
        details={
            "exception_type": type(exc).__name__
        },
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data
    )
