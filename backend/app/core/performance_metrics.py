"""
Performance metrics logging utilities.

This module provides utilities for tracking and logging performance metrics
including processing time, memory usage, and operation statistics.
"""

import time
import torch
from typing import Optional, Dict, Any, Callable
from functools import wraps
import inspect
from contextlib import contextmanager

from app.core.logging_config import get_context_logger


class PerformanceMetrics:
    """
    Container for performance metrics.
    """
    
    def __init__(self, operation_name: str):
        """
        Initialize performance metrics.
        
        Args:
            operation_name: Name of the operation being measured
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration_seconds = None
        self.memory_before = None
        self.memory_after = None
        self.memory_delta = None
        self.additional_metrics = {}
    
    def start(self) -> None:
        """Start timing and capture initial memory state."""
        self.start_time = time.time()
        self.memory_before = self._get_memory_stats()
    
    def stop(self) -> None:
        """Stop timing and capture final memory state."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.memory_after = self._get_memory_stats()
        self.memory_delta = self._calculate_memory_delta()
    
    def add_metric(self, key: str, value: Any) -> None:
        """
        Add an additional metric.
        
        Args:
            key: Metric name
            value: Metric value
        """
        self.additional_metrics[key] = value
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            "timestamp": time.time()
        }
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            stats.update({
                "gpu_allocated_mb": allocated / (1024 ** 2),
                "gpu_reserved_mb": reserved / (1024 ** 2),
                "gpu_total_mb": total / (1024 ** 2),
                "gpu_free_mb": (total - allocated) / (1024 ** 2),
                "gpu_usage_percent": (allocated / total * 100) if total > 0 else 0.0
            })
        
        return stats
    
    def _calculate_memory_delta(self) -> Dict[str, Any]:
        """Calculate memory usage delta."""
        if not self.memory_before or not self.memory_after:
            return {}
        
        delta = {}
        
        if "gpu_allocated_mb" in self.memory_before and "gpu_allocated_mb" in self.memory_after:
            delta["gpu_allocated_delta_mb"] = (
                self.memory_after["gpu_allocated_mb"] - self.memory_before["gpu_allocated_mb"]
            )
            delta["gpu_reserved_delta_mb"] = (
                self.memory_after["gpu_reserved_mb"] - self.memory_before["gpu_reserved_mb"]
            )
        
        return delta
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary for logging.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "operation": self.operation_name,
            "duration_seconds": round(self.duration_seconds, 3) if self.duration_seconds else None,
        }
        
        # Add memory metrics
        if self.memory_before:
            metrics["memory_before"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.memory_before.items()
                if k != "timestamp"
            }
        
        if self.memory_after:
            metrics["memory_after"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.memory_after.items()
                if k != "timestamp"
            }
        
        if self.memory_delta:
            metrics["memory_delta"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.memory_delta.items()
            }
        
        # Add additional metrics
        if self.additional_metrics:
            metrics["additional_metrics"] = self.additional_metrics
        
        return metrics
    
    def log(self, logger_name: str = "performance", request_id: Optional[str] = None) -> None:
        """
        Log the performance metrics.
        
        Args:
            logger_name: Name of the logger to use
            request_id: Request ID for tracking
        """
        logger = get_context_logger(logger_name, request_id)
        
        metrics_dict = self.to_dict()
        
        # Create summary message
        duration_str = f"{self.duration_seconds:.3f}s" if self.duration_seconds else "N/A"
        
        memory_summary = ""
        if self.memory_delta and "gpu_allocated_delta_mb" in self.memory_delta:
            delta = self.memory_delta["gpu_allocated_delta_mb"]
            memory_summary = f", Memory delta: {delta:+.2f}MB"
        
        message = f"Performance: {self.operation_name} completed in {duration_str}{memory_summary}"
        
        logger.info(message, extra={"performance_metrics": metrics_dict})


@contextmanager
def track_performance(
    operation_name: str,
    logger_name: str = "performance",
    request_id: Optional[str] = None,
    log_on_exit: bool = True
):
    """
    Context manager for tracking performance metrics.
    
    Usage:
        with track_performance("my_operation") as metrics:
            # code to measure
            metrics.add_metric("items_processed", 100)
    
    Args:
        operation_name: Name of the operation
        logger_name: Name of the logger to use
        request_id: Request ID for tracking
        log_on_exit: Whether to automatically log metrics on exit
        
    Yields:
        PerformanceMetrics instance
    """
    metrics = PerformanceMetrics(operation_name)
    metrics.start()
    
    try:
        yield metrics
    finally:
        metrics.stop()
        if log_on_exit:
            metrics.log(logger_name, request_id)


def measure_performance(
    operation_name: Optional[str] = None,
    logger_name: str = "performance",
    log_result: bool = True
):
    """
    Decorator to measure and log performance of a function.
    
    Usage:
        @measure_performance("my_operation")
        def my_function(arg1, arg2):
            # function code
            pass
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        logger_name: Name of the logger to use
        log_result: Whether to log the metrics
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = PerformanceMetrics(op_name)
            metrics.start()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                metrics.stop()
                if log_result:
                    metrics.log(logger_name)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = PerformanceMetrics(op_name)
            metrics.start()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics.stop()
                if log_result:
                    metrics.log(logger_name)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class OperationTimer:
    """
    Simple timer for measuring operation duration.
    
    Usage:
        timer = OperationTimer()
        # ... do work ...
        elapsed = timer.elapsed()
    """
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        return self.elapsed() * 1000
    
    def reset(self) -> float:
        """
        Reset timer and return elapsed time.
        
        Returns:
            Elapsed time before reset
        """
        elapsed = self.elapsed()
        self.start_time = time.time()
        return elapsed


class BatchMetrics:
    """
    Track metrics for batch operations.
    """
    
    def __init__(self, operation_name: str, batch_size: int):
        """
        Initialize batch metrics.
        
        Args:
            operation_name: Name of the batch operation
            batch_size: Number of items in the batch
        """
        self.operation_name = operation_name
        self.batch_size = batch_size
        self.start_time = None
        self.end_time = None
        self.duration_seconds = None
        self.items_processed = 0
        self.items_failed = 0
        self.item_times = []
    
    def start(self) -> None:
        """Start batch processing."""
        self.start_time = time.time()
    
    def record_item(self, success: bool, duration: float) -> None:
        """
        Record metrics for a single item.
        
        Args:
            success: Whether the item was processed successfully
            duration: Time taken to process the item
        """
        if success:
            self.items_processed += 1
        else:
            self.items_failed += 1
        
        self.item_times.append(duration)
    
    def stop(self) -> None:
        """Stop batch processing."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert batch metrics to dictionary.
        
        Returns:
            Dictionary with batch metrics
        """
        avg_item_time = sum(self.item_times) / len(self.item_times) if self.item_times else 0
        min_item_time = min(self.item_times) if self.item_times else 0
        max_item_time = max(self.item_times) if self.item_times else 0
        
        return {
            "operation": self.operation_name,
            "batch_size": self.batch_size,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "success_rate": (self.items_processed / self.batch_size * 100) if self.batch_size > 0 else 0,
            "total_duration_seconds": round(self.duration_seconds, 3) if self.duration_seconds else None,
            "avg_item_duration_seconds": round(avg_item_time, 3),
            "min_item_duration_seconds": round(min_item_time, 3),
            "max_item_duration_seconds": round(max_item_time, 3),
            "throughput_items_per_second": (
                self.items_processed / self.duration_seconds
                if self.duration_seconds and self.duration_seconds > 0
                else 0
            )
        }
    
    def log(self, logger_name: str = "performance", request_id: Optional[str] = None) -> None:
        """
        Log the batch metrics.
        
        Args:
            logger_name: Name of the logger to use
            request_id: Request ID for tracking
        """
        logger = get_context_logger(logger_name, request_id)
        
        metrics_dict = self.to_dict()
        
        message = (
            f"Batch Performance: {self.operation_name} - "
            f"{self.items_processed}/{self.batch_size} items processed in "
            f"{self.duration_seconds:.3f}s "
            f"({metrics_dict['throughput_items_per_second']:.2f} items/s)"
        )
        
        logger.info(message, extra={"batch_metrics": metrics_dict})


def log_memory_usage(
    operation_name: str,
    logger_name: str = "performance",
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Log current memory usage.
    
    Args:
        operation_name: Name of the operation
        logger_name: Name of the logger to use
        request_id: Request ID for tracking
        
    Returns:
        Dictionary with memory statistics
    """
    logger = get_context_logger(logger_name, request_id)
    
    memory_stats = {
        "operation": operation_name,
        "timestamp": time.time()
    }
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        
        memory_stats.update({
            "gpu_allocated_mb": round(allocated / (1024 ** 2), 2),
            "gpu_reserved_mb": round(reserved / (1024 ** 2), 2),
            "gpu_total_mb": round(total / (1024 ** 2), 2),
            "gpu_free_mb": round((total - allocated) / (1024 ** 2), 2),
            "gpu_usage_percent": round((allocated / total * 100) if total > 0 else 0.0, 2)
        })
        
        message = (
            f"Memory Usage ({operation_name}): "
            f"{memory_stats['gpu_allocated_mb']:.2f}MB / {memory_stats['gpu_total_mb']:.2f}MB "
            f"({memory_stats['gpu_usage_percent']:.1f}%)"
        )
    else:
        memory_stats["gpu_available"] = False
        message = f"Memory Usage ({operation_name}): GPU not available"
    
    logger.info(message, extra={"memory_stats": memory_stats})
    
    return memory_stats
