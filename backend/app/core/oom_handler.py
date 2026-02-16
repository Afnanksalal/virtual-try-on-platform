"""
Out-of-Memory (OOM) error handling utilities.

This module provides utilities for detecting, handling, and recovering from
GPU out-of-memory errors, including model unloading and CPU fallback.
"""

import torch
import gc
from typing import Callable, Any, Optional, List, TypeVar, ParamSpec
from functools import wraps
import inspect

from app.core.logging_config import get_context_logger
from app.core.exceptions import GPUOutOfMemoryException

logger = get_context_logger("oom_handler")

P = ParamSpec('P')
T = TypeVar('T')


def is_oom_error(exception: Exception) -> bool:
    """
    Check if an exception is an out-of-memory error.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if it's an OOM error, False otherwise
    """
    error_message = str(exception).lower()
    
    # CUDA OOM patterns
    cuda_oom_patterns = [
        "cuda out of memory",
        "cudnn error: cudnn_status_not_initialized",
        "cuda error: out of memory",
        "runtime error: cuda out of memory",
    ]
    
    # CPU OOM patterns
    cpu_oom_patterns = [
        "cannot allocate memory",
        "out of memory",
        "memoryerror",
    ]
    
    all_patterns = cuda_oom_patterns + cpu_oom_patterns
    
    return any(pattern in error_message for pattern in all_patterns)


def clear_gpu_memory() -> None:
    """
    Clear GPU memory by running garbage collection and emptying CUDA cache.
    """
    if torch.cuda.is_available():
        # Run garbage collection
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Synchronize to ensure operations complete
        torch.cuda.synchronize()
        
        logger.info("GPU memory cleared")


def get_gpu_memory_stats() -> dict:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    
    return {
        "available": True,
        "allocated_mb": allocated / (1024 ** 2),
        "reserved_mb": reserved / (1024 ** 2),
        "total_mb": total / (1024 ** 2),
        "free_mb": (total - allocated) / (1024 ** 2),
        "usage_percent": (allocated / total * 100) if total > 0 else 0.0
    }


def unload_models_for_memory(
    model_loader,
    models_to_keep: Optional[List[str]] = None,
    min_free_mb: float = 1000.0
) -> int:
    """
    Unload models to free GPU memory.
    
    Args:
        model_loader: ModelLoader instance
        models_to_keep: List of model names to keep loaded (others will be unloaded)
        min_free_mb: Minimum free memory in MB to achieve
        
    Returns:
        Number of models unloaded
    """
    if not torch.cuda.is_available():
        return 0
    
    models_to_keep = models_to_keep or []
    unloaded_count = 0
    
    # Get current memory stats
    memory_stats = get_gpu_memory_stats()
    current_free = memory_stats.get("free_mb", 0)
    
    logger.info(
        f"Current GPU memory: {memory_stats['allocated_mb']:.2f}MB allocated, "
        f"{current_free:.2f}MB free"
    )
    
    # If we already have enough free memory, no need to unload
    if current_free >= min_free_mb:
        logger.info(f"Sufficient free memory available ({current_free:.2f}MB >= {min_free_mb}MB)")
        return 0
    
    # Get list of loaded models
    loaded_models = list(model_loader.models.keys())
    
    # Unload models (except those in keep list)
    for model_name in loaded_models:
        if model_name in models_to_keep:
            continue
        
        logger.info(f"Unloading model '{model_name}' to free memory")
        model_loader.unload_model(model_name)
        unloaded_count += 1
        
        # Check if we have enough free memory now
        clear_gpu_memory()
        memory_stats = get_gpu_memory_stats()
        current_free = memory_stats.get("free_mb", 0)
        
        logger.info(f"After unloading '{model_name}': {current_free:.2f}MB free")
        
        if current_free >= min_free_mb:
            logger.info(f"Target free memory achieved ({current_free:.2f}MB >= {min_free_mb}MB)")
            break
    
    return unloaded_count


def handle_oom_error(
    exception: Exception,
    model_loader,
    operation_name: str,
    retry_on_cpu: bool = True,
    models_to_keep: Optional[List[str]] = None
) -> dict:
    """
    Handle an out-of-memory error by attempting recovery strategies.
    
    Args:
        exception: The OOM exception
        model_loader: ModelLoader instance
        operation_name: Name of the operation that failed
        retry_on_cpu: Whether to suggest CPU fallback
        models_to_keep: Models to keep loaded during cleanup
        
    Returns:
        Dictionary with recovery information
    """
    logger.error(f"OOM error during {operation_name}: {str(exception)}")
    
    # Get memory stats before cleanup
    memory_before = get_gpu_memory_stats()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Unload models to free memory
    unloaded_count = unload_models_for_memory(
        model_loader,
        models_to_keep=models_to_keep,
        min_free_mb=1500.0  # Try to free at least 1.5GB
    )
    
    # Get memory stats after cleanup
    memory_after = get_gpu_memory_stats()
    
    recovery_info = {
        "oom_detected": True,
        "operation": operation_name,
        "models_unloaded": unloaded_count,
        "memory_before_mb": memory_before.get("allocated_mb", 0),
        "memory_after_mb": memory_after.get("allocated_mb", 0),
        "memory_freed_mb": memory_before.get("allocated_mb", 0) - memory_after.get("allocated_mb", 0),
        "retry_recommended": unloaded_count > 0,
        "cpu_fallback_available": retry_on_cpu and torch.cuda.is_available(),
    }
    
    logger.info(
        f"OOM recovery: unloaded {unloaded_count} models, "
        f"freed {recovery_info['memory_freed_mb']:.2f}MB"
    )
    
    return recovery_info


def with_oom_handling(
    model_loader,
    max_retries: int = 2,
    retry_on_cpu: bool = True,
    models_to_keep: Optional[List[str]] = None
):
    """
    Decorator to add OOM error handling to a function.
    
    The decorator will:
    1. Catch OOM errors
    2. Unload models to free memory
    3. Retry the operation
    4. Fall back to CPU if GPU retries fail (if enabled)
    
    Usage:
        @with_oom_handling(model_loader, max_retries=2, retry_on_cpu=True)
        def my_gpu_operation(data):
            # GPU-intensive code
            pass
    
    Args:
        model_loader: ModelLoader instance
        max_retries: Maximum number of retry attempts after OOM
        retry_on_cpu: Whether to fall back to CPU after GPU retries fail
        models_to_keep: Models to keep loaded during cleanup
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            last_exception = None
            
            while attempt <= max_retries:
                try:
                    # Try to execute the function
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if it's an OOM error
                    if not is_oom_error(e):
                        # Not an OOM error, re-raise immediately
                        raise
                    
                    attempt += 1
                    logger.warning(
                        f"OOM error in {func.__name__} (attempt {attempt}/{max_retries + 1})"
                    )
                    
                    if attempt <= max_retries:
                        # Try to recover
                        recovery_info = handle_oom_error(
                            exception=e,
                            model_loader=model_loader,
                            operation_name=func.__name__,
                            retry_on_cpu=retry_on_cpu,
                            models_to_keep=models_to_keep
                        )
                        
                        if recovery_info["retry_recommended"]:
                            logger.info(f"Retrying {func.__name__} after OOM recovery...")
                            continue
                    
                    # Max retries reached or recovery not possible
                    if retry_on_cpu and torch.cuda.is_available():
                        logger.warning(
                            f"GPU retries exhausted for {func.__name__}. "
                            "CPU fallback may be available but not implemented in this decorator."
                        )
                    
                    # Raise GPUOutOfMemoryException with context
                    memory_stats = get_gpu_memory_stats()
                    raise GPUOutOfMemoryException(
                        details={
                            "function": func.__name__,
                            "attempts": attempt,
                            "memory_stats": memory_stats,
                            "original_error": str(last_exception)
                        }
                    ) from last_exception
            
            # Should not reach here, but just in case
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            last_exception = None
            
            while attempt <= max_retries:
                try:
                    # Try to execute the function
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if it's an OOM error
                    if not is_oom_error(e):
                        # Not an OOM error, re-raise immediately
                        raise
                    
                    attempt += 1
                    logger.warning(
                        f"OOM error in {func.__name__} (attempt {attempt}/{max_retries + 1})"
                    )
                    
                    if attempt <= max_retries:
                        # Try to recover
                        recovery_info = handle_oom_error(
                            exception=e,
                            model_loader=model_loader,
                            operation_name=func.__name__,
                            retry_on_cpu=retry_on_cpu,
                            models_to_keep=models_to_keep
                        )
                        
                        if recovery_info["retry_recommended"]:
                            logger.info(f"Retrying {func.__name__} after OOM recovery...")
                            continue
                    
                    # Max retries reached or recovery not possible
                    if retry_on_cpu and torch.cuda.is_available():
                        logger.warning(
                            f"GPU retries exhausted for {func.__name__}. "
                            "CPU fallback may be available but not implemented in this decorator."
                        )
                    
                    # Raise GPUOutOfMemoryException with context
                    memory_stats = get_gpu_memory_stats()
                    raise GPUOutOfMemoryException(
                        details={
                            "function": func.__name__,
                            "attempts": attempt,
                            "memory_stats": memory_stats,
                            "original_error": str(last_exception)
                        }
                    ) from last_exception
            
            # Should not reach here, but just in case
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class OOMContextManager:
    """
    Context manager for handling OOM errors within a code block.
    
    Usage:
        with OOMContextManager(model_loader, operation_name="my_operation"):
            # GPU-intensive code
            pass
    """
    
    def __init__(
        self,
        model_loader,
        operation_name: str,
        models_to_keep: Optional[List[str]] = None,
        raise_on_oom: bool = True
    ):
        """
        Initialize OOM context manager.
        
        Args:
            model_loader: ModelLoader instance
            operation_name: Name of the operation
            models_to_keep: Models to keep loaded during cleanup
            raise_on_oom: Whether to re-raise OOM exceptions after handling
        """
        self.model_loader = model_loader
        self.operation_name = operation_name
        self.models_to_keep = models_to_keep
        self.raise_on_oom = raise_on_oom
        self.recovery_info = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None and is_oom_error(exc_value):
            # Handle OOM error
            self.recovery_info = handle_oom_error(
                exception=exc_value,
                model_loader=self.model_loader,
                operation_name=self.operation_name,
                models_to_keep=self.models_to_keep
            )
            
            # Suppress or re-raise based on configuration
            return not self.raise_on_oom
        
        # Don't suppress other exceptions
        return False
