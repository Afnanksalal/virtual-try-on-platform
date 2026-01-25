"""
Centralized logging configuration for production-ready logging.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Optional
import json
from datetime import datetime
from contextvars import ContextVar
import traceback

# Context variable for request ID propagation across async boundaries
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging with required fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Required fields for all log entries
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": self._get_request_id(record),
        }
        
        # Additional context fields
        log_data.update({
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
        })
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack_trace": self.formatException(record.exc_info)
            }
        
        # Add any extra fields passed via extra parameter
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'request_id']:
                try:
                    # Only add JSON-serializable values
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        return json.dumps(log_data)
    
    def _get_request_id(self, record: logging.LogRecord) -> Optional[str]:
        """Get request ID from record or context."""
        # First check if request_id is in the log record
        if hasattr(record, "request_id"):
            return record.request_id
        
        # Fall back to context variable
        return request_id_context.get()


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Configure application logging with structured JSON format.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("vton")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatting for production
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name suffix (will be prefixed with 'vton.')
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"vton.{name}")
    return logging.getLogger("vton")


def set_request_id(request_id: str) -> None:
    """
    Set the request ID in the context for propagation across async calls.
    
    Args:
        request_id: The request ID to set
    """
    request_id_context.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Get the current request ID from context.
    
    Returns:
        Current request ID or None if not set
    """
    return request_id_context.get()


def clear_request_id() -> None:
    """Clear the request ID from context."""
    request_id_context.set(None)


class RequestContextLogger:
    """
    Logger wrapper that automatically includes request ID in all log calls.
    """
    
    def __init__(self, logger: logging.Logger, request_id: Optional[str] = None):
        """
        Initialize the context logger.
        
        Args:
            logger: The underlying logger instance
            request_id: Optional request ID (will use context if not provided)
        """
        self.logger = logger
        self._request_id = request_id
    
    @property
    def request_id(self) -> Optional[str]:
        """Get the request ID (from instance or context)."""
        return self._request_id or get_request_id()
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal logging method that adds request_id to extra."""
        extra = kwargs.get('extra', {})
        if self.request_id:
            extra['request_id'] = self.request_id
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with request context."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message with request context."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with request context."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message with request context."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with request context."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with request context."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)


def get_context_logger(name: str = None, request_id: Optional[str] = None) -> RequestContextLogger:
    """
    Get a context-aware logger that automatically includes request ID.
    
    Args:
        name: Optional logger name suffix
        request_id: Optional request ID (will use context if not provided)
    
    Returns:
        RequestContextLogger instance
    """
    logger = get_logger(name)
    return RequestContextLogger(logger, request_id)
