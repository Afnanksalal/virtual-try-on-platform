"""
Custom middleware for request logging, rate limiting, and error tracking.
"""
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
from typing import Callable, Optional
from app.core.logging_config import get_logger, set_request_id, clear_request_id, get_context_logger
from supabase import create_client, Client
import os

logger = get_logger("middleware")


async def get_user_from_token(authorization: Optional[str]) -> Optional[str]:
    """
    Extract and verify user ID from Supabase JWT token.
    
    Args:
        authorization: Authorization header value (Bearer token)
        
    Returns:
        User ID if valid token, None otherwise
    """
    if not authorization:
        return None
    
    if not authorization.startswith("Bearer "):
        return None
    
    token = authorization.replace("Bearer ", "")
    
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase not configured for auth verification")
            return None
        
        client: Client = create_client(supabase_url, supabase_key)
        
        # Verify token and get user
        user_response = client.auth.get_user(token)
        
        if user_response and user_response.user:
            return user_response.user.id
        
        return None
        
    except Exception as e:
        logger.debug(f"Token verification failed: {e}")
        return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests with timing information and request ID propagation."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract user ID from Authorization header
        authorization = request.headers.get("Authorization")
        user_id = await get_user_from_token(authorization)
        request.state.user_id = user_id
        
        # Set request ID in context for propagation
        set_request_id(request_id)
        
        # Get context-aware logger
        ctx_logger = get_context_logger("middleware", request_id)
        
        # Log request start
        start_time = time.time()
        
        log_extra = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        if user_id:
            log_extra["user_id"] = user_id
            ctx_logger.info(f"Request started: {request.method} {request.url.path} (User: {user_id})", extra=log_extra)
        else:
            ctx_logger.info(f"Request started: {request.method} {request.url.path}", extra=log_extra)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request completion
            log_extra.update({
                "status_code": response.status_code,
                "duration_seconds": round(duration, 3),
            })
            
            ctx_logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Duration: {duration:.3f}s",
                extra=log_extra
            )
            
            # Add request ID and user ID to response headers
            response.headers["X-Request-ID"] = request_id
            if user_id:
                response.headers["X-User-ID"] = user_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            ctx_logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Duration: {duration:.3f}s",
                extra={
                    **log_extra,
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
