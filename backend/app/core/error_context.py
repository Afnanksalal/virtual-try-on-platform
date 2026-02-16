"""
Error context utilities for capturing and logging error details.

This module provides utilities for capturing input parameters, stack traces,
and other contextual information when errors occur.
"""

import traceback
import sys
from typing import Any, Dict, Optional, Callable
from functools import wraps
import inspect
import json

from app.core.logging_config import get_context_logger


def sanitize_for_logging(value: Any, max_length: int = 1000) -> Any:
    """
    Sanitize a value for safe logging (remove sensitive data, truncate large values).
    
    Args:
        value: Value to sanitize
        max_length: Maximum string length before truncation
        
    Returns:
        Sanitized value safe for logging
    """
    # Handle None
    if value is None:
        return None
    
    # Handle strings
    if isinstance(value, str):
        # Check for potential sensitive data patterns
        lower_value = value.lower()
        if any(keyword in lower_value for keyword in ['password', 'token', 'secret', 'key', 'auth']):
            return "[REDACTED]"
        
        # Truncate long strings
        if len(value) > max_length:
            return value[:max_length] + f"... (truncated, total length: {len(value)})"
        
        return value
    
    # Handle bytes
    if isinstance(value, bytes):
        return f"<bytes: {len(value)} bytes>"
    
    # Handle dictionaries
    if isinstance(value, dict):
        return {k: sanitize_for_logging(v, max_length) for k, v in value.items()}
    
    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        sanitized = [sanitize_for_logging(item, max_length) for item in value[:10]]
        if len(value) > 10:
            sanitized.append(f"... ({len(value) - 10} more items)")
        return sanitized
    
    # Handle other types
    try:
        # Try to convert to string
        str_value = str(value)
        if len(str_value) > max_length:
            return str_value[:max_length] + "... (truncated)"
        return str_value
    except Exception:
        return f"<{type(value).__name__}>"


def capture_function_args(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Capture function arguments for error logging.
    
    Args:
        func: The function being called
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Dictionary of argument names and sanitized values
    """
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Sanitize all arguments
        sanitized_args = {
            name: sanitize_for_logging(value)
            for name, value in bound_args.arguments.items()
        }
        
        return sanitized_args
    except Exception as e:
        return {"error_capturing_args": str(e)}


def get_exception_context(exc: Exception) -> Dict[str, Any]:
    """
    Extract detailed context from an exception.
    
    Args:
        exc: The exception to extract context from
        
    Returns:
        Dictionary with exception details
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    context = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_module": type(exc).__module__,
    }
    
    # Add stack trace
    if exc_traceback:
        tb_lines = traceback.format_tb(exc_traceback)
        context["stack_trace"] = "".join(tb_lines)
        
        # Extract the last frame for quick reference
        tb_frames = traceback.extract_tb(exc_traceback)
        if tb_frames:
            last_frame = tb_frames[-1]
            context["error_location"] = {
                "file": last_frame.filename,
                "line": last_frame.lineno,
                "function": last_frame.name,
                "code": last_frame.line
            }
    
    # Add exception attributes if available
    if hasattr(exc, '__dict__'):
        exc_attrs = {
            k: sanitize_for_logging(v)
            for k, v in exc.__dict__.items()
            if not k.startswith('_')
        }
        if exc_attrs:
            context["exception_attributes"] = exc_attrs
    
    return context


def log_error_with_context(
    logger_name: str,
    error_message: str,
    exception: Optional[Exception] = None,
    input_params: Optional[Dict[str, Any]] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log an error with full context including stack trace and input parameters.
    
    Args:
        logger_name: Name of the logger to use
        error_message: Human-readable error message
        exception: The exception that occurred (if any)
        input_params: Input parameters that led to the error
        additional_context: Any additional context to include
        request_id: Request ID for tracking
    """
    logger = get_context_logger(logger_name, request_id)
    
    # Build log context
    log_context = {}
    
    # Add input parameters
    if input_params:
        log_context["input_params"] = sanitize_for_logging(input_params)
    
    # Add exception context
    if exception:
        log_context.update(get_exception_context(exception))
    
    # Add additional context
    if additional_context:
        log_context["additional_context"] = sanitize_for_logging(additional_context)
    
    # Log the error
    logger.error(
        error_message,
        extra=log_context,
        exc_info=exception is not None
    )


def error_context(logger_name: str = None):
    """
    Decorator to automatically log errors with full context.
    
    Usage:
        @error_context("my_module")
        def my_function(arg1, arg2):
            # function code
            pass
    
    Args:
        logger_name: Name of the logger to use (defaults to function's module)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Capture function arguments
                func_args = capture_function_args(func, args, kwargs)
                
                # Determine logger name
                log_name = logger_name or func.__module__
                
                # Log error with context
                log_error_with_context(
                    logger_name=log_name,
                    error_message=f"Error in {func.__name__}: {str(e)}",
                    exception=e,
                    input_params=func_args,
                    additional_context={
                        "function": func.__name__,
                        "module": func.__module__,
                    }
                )
                
                # Re-raise the exception
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Capture function arguments
                func_args = capture_function_args(func, args, kwargs)
                
                # Determine logger name
                log_name = logger_name or func.__module__
                
                # Log error with context
                log_error_with_context(
                    logger_name=log_name,
                    error_message=f"Error in {func.__name__}: {str(e)}",
                    exception=e,
                    input_params=func_args,
                    additional_context={
                        "function": func.__name__,
                        "module": func.__module__,
                    }
                )
                
                # Re-raise the exception
                raise
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ErrorContextManager:
    """
    Context manager for capturing and logging errors with context.
    
    Usage:
        with ErrorContextManager("my_operation", input_data=data):
            # code that might raise errors
            pass
    """
    
    def __init__(
        self,
        operation_name: str,
        logger_name: str = "error_context",
        request_id: Optional[str] = None,
        **context_kwargs
    ):
        """
        Initialize error context manager.
        
        Args:
            operation_name: Name of the operation being performed
            logger_name: Name of the logger to use
            request_id: Request ID for tracking
            **context_kwargs: Additional context to capture
        """
        self.operation_name = operation_name
        self.logger_name = logger_name
        self.request_id = request_id
        self.context = context_kwargs
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            # An exception occurred, log it with context
            log_error_with_context(
                logger_name=self.logger_name,
                error_message=f"Error during {self.operation_name}: {str(exc_value)}",
                exception=exc_value,
                additional_context={
                    "operation": self.operation_name,
                    **self.context
                },
                request_id=self.request_id
            )
        
        # Don't suppress the exception
        return False
