# memory_system/utils/exceptions.py
"""Comprehensive exception hierarchy for Unified Memory System.

This module provides a complete exception system with:
- Structured error handling with JSON serialization
- Hierarchical exception types for different error categories  
- Context information and cause chaining for debugging
- Helper functions and decorators for exception handling
- Integration with logging and API error responses

The exception hierarchy is designed to be:
- Hierarchical: Specific exceptions inherit from more general ones
- Serializable: All exceptions can be converted to JSON
- Contextual: Exceptions carry additional debugging information
- Traceable: Support for cause chaining and error propagation
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, Type, TypeVar

__all__ = [
    # Base exception
    "MemorySystemError",
    
    # Validation and configuration errors
    "ValidationError",
    "ConfigurationError",
    
    # Storage and database errors
    "StorageError", 
    "DatabaseError",
    
    # ML and processing errors
    "EmbeddingError",
    
    # API and network errors
    "APIError",
    "RateLimitError",
    "TimeoutError",
    
    # Resource and system errors
    "ResourceError",
    
    # Security errors
    "SecurityError",
    "AuthenticationError", 
    "AuthorizationError",
    
    # Helper functions
    "wrap_exception",
    "create_validation_error",
    "log_exception",
]

log = logging.getLogger(__name__)
_T = TypeVar("_T", bound="MemorySystemError")


# =============================================================================
# Base Exception Class
# =============================================================================

class MemorySystemError(RuntimeError):
    """Base exception class for all Unified Memory System errors.
    
    This exception provides comprehensive error handling with:
    - JSON serialization for structured logging and API responses
    - Context information storage for debugging
    - Cause chain support for error tracing
    - Timestamp tracking for forensic analysis
    - Configurable error codes for programmatic handling
    
    Attributes:
        message: Human-readable error description
        context: Additional context information as key-value pairs
        code: Error code for programmatic handling
        ts_utc: UTC timestamp when the error occurred
        
    Example:
        try:
            risky_operation()
        except Exception as e:
            raise MemorySystemError(
                "Operation failed",
                context={"operation": "risky_operation", "retry_count": 3},
                cause=e
            )
    """
    
    default_code: str = "memory_system_error"

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        """Initialize a new MemorySystemError.
        
        Args:
            message: Human-readable error description
            context: Additional context information for debugging
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.code = self.default_code
        self.ts_utc = dt.datetime.utcnow()
        
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a JSON-serializable dictionary.
        
        This method creates a structured representation of the exception
        that can be used for logging, API responses, or error reporting.
        
        Returns:
            Dictionary containing all exception information including:
            - error: The error code
            - message: Human-readable description
            - timestamp: ISO format timestamp
            - context: Additional context information
            - cause: Information about the underlying cause (if any)
        """
        payload: Dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "timestamp": self.ts_utc.isoformat() + "Z",
        }
        
        if self.context:
            payload["context"] = self.context
            
        if self.__cause__ is not None:
            payload["cause"] = {
                "type": type(self.__cause__).__name__,
                "message": str(self.__cause__),
            }
            
        return payload

    def __str__(self) -> str:
        """Return compact JSON representation of the exception."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


# =============================================================================
# Validation and Configuration Errors
# =============================================================================

class ValidationError(MemorySystemError):
    """Exception raised when input data fails validation.
    
    This exception is used for:
    - Invalid request payloads and parameters
    - Schema validation failures  
    - Data type and format errors
    - Constraint violations
    - Business rule violations
    
    Example:
        if not email_pattern.match(email):
            raise ValidationError(
                "Invalid email format",
                context={"email": email, "expected_pattern": "user@domain.com"}
            )
    """
    
    default_code = "validation_error"


class ConfigurationError(MemorySystemError):
    """Exception raised when system configuration is invalid or missing.
    
    This exception is used for:
    - Missing required configuration values
    - Invalid configuration file formats
    - Conflicting configuration settings
    - Environment setup errors
    - Dependency configuration issues
    
    Example:
        if not settings.database_url:
            raise ConfigurationError(
                "Database URL not configured",
                context={"config_file": "settings.yaml", "required_field": "database_url"}
            )
    """
    
    default_code = "configuration_error"


# =============================================================================
# Storage and Database Errors  
# =============================================================================

class StorageError(MemorySystemError):
    """Exception raised for general storage layer failures.
    
    This is a base class for all storage-related errors including
    database connections, file system operations, and data persistence.
    More specific storage errors should inherit from this class.
    """
    
    default_code = "storage_error"


class DatabaseError(StorageError):
    """Exception raised for database-specific failures.
    
    This exception is used for:
    - SQL execution errors and syntax issues
    - Database connection failures
    - Transaction rollback errors
    - Schema migration failures
    - Data integrity constraint violations
    - Connection pool exhaustion
    
    Example:
        try:
            cursor.execute(sql, params)
        except sqlite3.Error as e:
            raise DatabaseError(
                "SQL query execution failed",
                context={"sql": sql, "params": params},
                cause=e
            )
    """
    
    default_code = "database_error"


# =============================================================================
# Machine Learning and Processing Errors
# =============================================================================

class EmbeddingError(MemorySystemError):
    """Exception raised for text embedding and model inference issues.
    
    This exception is used for:
    - Model loading and initialization failures
    - Embedding generation errors
    - Vector dimension mismatches
    - Model inference timeouts
    - CUDA/GPU memory errors
    - Model format compatibility issues
    
    Example:
        try:
            embeddings = model.encode(texts)
        except Exception as e:
            raise EmbeddingError(
                "Failed to generate text embeddings",
                context={"model": model_name, "text_count": len(texts)},
                cause=e
            )
    """
    
    default_code = "embedding_error"


# =============================================================================
# API and Network Errors
# =============================================================================

class APIError(MemorySystemError):
    """Exception raised for general API layer errors.
    
    This exception is used for:
    - Request processing failures
    - Response serialization errors
    - Middleware processing errors
    - Internal server errors
    - Service unavailability
    
    Example:
        try:
            response = await process_request(request)
        except Exception as e:
            raise APIError(
                "Request processing failed",
                context={"endpoint": request.url.path, "method": request.method},
                cause=e
            )
    """
    
    default_code = "api_error"


class RateLimitError(MemorySystemError):
    """Exception raised when rate limits are exceeded.
    
    This exception corresponds to HTTP 429 Too Many Requests and is used
    for request throttling, abuse prevention, and resource protection.
    
    Example:
        if request_count > limit:
            raise RateLimitError(
                "Rate limit exceeded",
                context={
                    "user_id": user_id,
                    "requests": request_count,
                    "limit": limit,
                    "window_seconds": window
                }
            )
    """
    
    default_code = "rate_limit_error"


class TimeoutError(MemorySystemError):
    """Exception raised when operations exceed allowed time limits.
    
    This exception is used for:
    - HTTP request timeouts
    - Database query timeouts  
    - Model inference timeouts
    - Network operation timeouts
    - Background task timeouts
    
    Example:
        try:
            result = await asyncio.wait_for(operation(), timeout=30)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                "Operation timed out",
                context={"timeout_seconds": 30, "operation": "model_inference"},
                cause=e
            )
    """
    
    default_code = "timeout_error"


# =============================================================================
# Resource and System Errors
# =============================================================================

class ResourceError(MemorySystemError):
    """Exception raised for resource availability and exhaustion issues.
    
    This exception is used for:
    - Out of memory conditions
    - Disk space exhaustion
    - File descriptor limits
    - CPU/GPU resource exhaustion
    - Network bandwidth limits
    - Connection pool exhaustion
    
    Example:
        if available_memory < required_memory:
            raise ResourceError(
                "Insufficient memory for operation",
                context={
                    "available_mb": available_memory // 1024**2,
                    "required_mb": required_memory // 1024**2
                }
            )
    """
    
    default_code = "resource_error"


# =============================================================================
# Security Errors
# =============================================================================

class SecurityError(MemorySystemError):
    """Exception raised for general security violations.
    
    This is a base class for all security-related errors including
    authentication, authorization, and data protection issues.
    """
    
    default_code = "security_error"


class AuthenticationError(SecurityError):
    """Exception raised for authentication failures.
    
    This exception is used for:
    - Invalid credentials (username/password)
    - Expired or invalid tokens
    - Missing authentication headers
    - Token signature verification failures
    - Multi-factor authentication failures
    
    Example:
        if not verify_token(token):
            raise AuthenticationError(
                "Invalid authentication token",
                context={"token_type": "bearer", "expired": is_expired(token)}
            )
    """
    
    default_code = "authentication_error"


class AuthorizationError(SecurityError):
    """Exception raised for authorization and permission failures.
    
    This exception is used for:
    - Insufficient permissions for resources
    - Role-based access control violations
    - Resource ownership violations
    - Scope and permission mismatches
    - Access policy violations
    
    Example:
        if not user.has_permission("write", resource):
            raise AuthorizationError(
                "Insufficient permissions",
                context={
                    "user_id": user.id,
                    "required_permission": "write",
                    "resource": resource.id
                }
            )
    """
    
    default_code = "authorization_error"


# =============================================================================
# Helper Functions and Decorators
# =============================================================================

def wrap_exception(exc_type: Type[_T], message: str):
    """Decorator to wrap any exception in a specific MemorySystemError type.
    
    This decorator catches any exception raised by the decorated function
    and wraps it in the specified exception type while preserving the
    original exception as the cause. It automatically handles both
    synchronous and asynchronous functions.
    
    Args:
        exc_type: The MemorySystemError subclass to wrap exceptions in
        message: The message to use for the wrapper exception
        
    Returns:
        Decorated function that wraps exceptions in the specified type
        
    Example:
        @wrap_exception(DatabaseError, "Failed to execute database query")
        async def execute_query(sql: str, params: tuple):
            # Any exception here will be wrapped in DatabaseError
            cursor.execute(sql, params)
            return cursor.fetchall()
    """
    def decorator(func: Callable) -> Callable:
        async def _inner_async(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except MemorySystemError:
                # Don't wrap our own exceptions to avoid double-wrapping
                raise
            except Exception as e:
                raise exc_type(message, cause=e) from e

        def _inner_sync(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MemorySystemError:
                # Don't wrap our own exceptions to avoid double-wrapping
                raise
            except Exception as e:
                raise exc_type(message, cause=e) from e

        # Return appropriate wrapper based on function type
        return _inner_async if asyncio.iscoroutinefunction(func) else _inner_sync

    return decorator


def create_validation_error(field: str, detail: str, **context) -> ValidationError:
    """Create a ValidationError with standardized field information.
    
    This is a convenience function for creating validation errors with
    consistent structure and context information. It automatically
    includes field and detail information in the context.
    
    Args:
        field: The name of the field that failed validation
        detail: Detailed description of the validation failure
        **context: Additional context information to include
        
    Returns:
        ValidationError with structured context information
        
    Example:
        # Simple field validation
        error = create_validation_error("email", "Invalid email format")
        
        # With additional context
        error = create_validation_error(
            "age",
            "Must be between 18 and 120",
            value=15,
            min_value=18,
            max_value=120
        )
    """
    context["field"] = field
    context["detail"] = detail
    return ValidationError(
        f"Validation failed for field '{field}': {detail}", 
        context=context
    )


def log_exception(
    exception: MemorySystemError, 
    *, 
    level: int = logging.ERROR,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log a MemorySystemError with structured JSON payload.
    
    This function logs exceptions in a structured format that's easy
    to parse and analyze in log aggregation systems like ELK stack,
    Splunk, or cloud logging services.
    
    Args:
        exception: The MemorySystemError to log
        level: Logging level to use (default: ERROR)
        logger: Logger instance to use (default: module logger)
        
    Example:
        try:
            risky_database_operation()
        except Exception as e:
            error = DatabaseError("Database operation failed", cause=e)
            log_exception(error, level=logging.WARNING)
            
        # Custom logger
        custom_logger = logging.getLogger("myapp.database")
        log_exception(error, logger=custom_logger)
    """
    if logger is None:
        logger = log
        
    # Use structured logging with JSON format
    logger.log(level, "%s", json.dumps(exception.to_dict(), ensure_ascii=False))
