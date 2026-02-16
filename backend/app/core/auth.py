"""
Authentication utilities for validating Supabase JWT tokens.

This module provides FastAPI dependencies for protecting endpoints
and extracting authenticated user information from JWT tokens.
"""

from fastapi import Depends, HTTPException, Header
from typing import Optional
from app.core.logging_config import get_logger
from supabase import create_client, Client
import os

logger = get_logger("auth")


async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency to extract and validate user ID from Supabase JWT token.
    
    This dependency should be used on all protected endpoints that require authentication.
    It validates the JWT token and returns the authenticated user's ID.
    
    Args:
        authorization: Authorization header value (Bearer token)
        
    Returns:
        User ID string if authentication succeeds
        
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
        
    Example:
        @router.get("/protected")
        async def protected_endpoint(user_id: str = Depends(get_current_user)):
            # user_id is now available and validated
            return {"user_id": user_id}
    """
    if not authorization:
        logger.warning("Authentication failed: No authorization header provided")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "UNAUTHORIZED",
                    "message": "Authentication required. Please provide a valid authorization token.",
                    "details": {"reason": "Missing Authorization header"}
                }
            }
        )
    
    if not authorization.startswith("Bearer "):
        logger.warning("Authentication failed: Invalid authorization header format")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "UNAUTHORIZED",
                    "message": "Invalid authorization format. Expected 'Bearer <token>'.",
                    "details": {"reason": "Invalid Authorization header format"}
                }
            }
        )
    
    token = authorization.replace("Bearer ", "")
    
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase not configured for authentication")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "type": "system_error",
                        "code": "AUTH_CONFIG_ERROR",
                        "message": "Authentication service is not properly configured.",
                        "details": {"reason": "Supabase credentials missing"}
                    }
                }
            )
        
        client: Client = create_client(supabase_url, supabase_key)
        
        # Verify token and get user
        user_response = client.auth.get_user(token)
        
        if not user_response or not user_response.user:
            logger.warning("Authentication failed: Invalid or expired token")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "type": "user_error",
                        "code": "UNAUTHORIZED",
                        "message": "Invalid or expired authentication token. Please log in again.",
                        "details": {"reason": "Token verification failed"}
                    }
                }
            )
        
        user_id = user_response.user.id
        logger.debug(f"User authenticated successfully: {user_id}")
        
        return user_id
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "UNAUTHORIZED",
                    "message": "Authentication failed. Please log in again.",
                    "details": {"reason": str(e)}
                }
            }
        )


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    FastAPI dependency to optionally extract user ID from Supabase JWT token.
    
    This dependency is for endpoints that can work with or without authentication.
    If a valid token is provided, it returns the user ID. Otherwise, returns None.
    
    Args:
        authorization: Authorization header value (Bearer token)
        
    Returns:
        User ID string if valid token provided, None otherwise
        
    Example:
        @router.get("/public-or-private")
        async def flexible_endpoint(user_id: Optional[str] = Depends(get_optional_user)):
            if user_id:
                # Authenticated user - provide personalized response
                return {"user_id": user_id, "personalized": True}
            else:
                # Anonymous user - provide generic response
                return {"personalized": False}
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
            logger.warning("Supabase not configured for optional authentication")
            return None
        
        client: Client = create_client(supabase_url, supabase_key)
        
        # Verify token and get user
        user_response = client.auth.get_user(token)
        
        if user_response and user_response.user:
            user_id = user_response.user.id
            logger.debug(f"Optional auth: User authenticated: {user_id}")
            return user_id
        
        return None
        
    except Exception as e:
        logger.debug(f"Optional auth failed (non-critical): {e}")
        return None
