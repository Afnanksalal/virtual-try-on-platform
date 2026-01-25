"""
Custom middleware for request logging, rate limiting, and error tracking.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
from typing import Callable
from app.core.logging_config import get_logger, set_request_id, clear_request_id, get_context_logger

logger = get_logger("middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests with timing information and request ID propagation."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Set request ID in context for propagation
        set_request_id(request_id)
        
        # Get context-aware logger
        ctx_logger = get_context_logger("middleware", request_id)
        
        # Log request start
        start_time = time.time()
        ctx_logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request completion
            ctx_logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Duration: {duration:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": round(duration, 3),
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            ctx_logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Duration: {duration:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_seconds": round(duration, 3),
                    "error_type": type(e).__name__,
                },
                exc_info=True
            )
            raise
        finally:
            # Clear request ID from context after request completes
            clear_request_id()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not hasattr(request.state, "request_id"):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            set_request_id(request_id)
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
