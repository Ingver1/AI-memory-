"""Comprehensive tests for API module."""

import asyncio
import json
import time
from typing import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_system.api.app import create_app
from memory_system.api.schemas import (
    HealthResponse,
    StatsResponse,
    MemoryCreate,
    MemoryRead,
    MemoryQuery,
    MemorySearchResult,
    SuccessResponse,
    ErrorResponse,
)
from memory_system.config.settings import UnifiedSettings


@pytest.fixture
def test_settings():
    """Create test settings."""
    return UnifiedSettings.for_testing()


@pytest.fixture
def test_app(test_settings):
    """Create test FastAPI app."""
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
async def async_test_client(test_app):
    """Create async test client."""
    async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Unified Memory System"
        assert data["version"] == "0.8-alpha"
        assert data["status"] == "running"
        assert data["documentation"] == "/docs"
        assert data["health"] == "/health"
        assert data["metrics"] == "/metrics"
        assert data["api_version"] == "v1"

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        health_response = HealthResponse(**data)
        
        assert health_response.status in ["healthy", "degraded", "unhealthy"]
        assert health_response.version == "0.8-alpha"
        assert health_response.uptime_seconds >= 0
        assert isinstance(health_response.checks, dict)
        assert isinstance(health_response.memory_store_health, dict)
        assert isinstance(health_response.api_enabled, bool)
        assert health_response.timestamp is not None

    def test_liveness_probe(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/api/v1/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_readiness_probe(self, test_client):
        """Test readiness probe endpoint."""
        response = test_client.get("/api/v1/health/ready")
        # Should be 200 for healthy service or 503 for unhealthy
        assert response.status_code in [200, 503]
        
        data = response.json()
        if response.status_code == 200:
            assert data["status"] == "ready"
        else:
            assert "detail" in data

    def test_stats_endpoint(self, test_client):
        """Test stats endpoint."""
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        stats_response = StatsResponse(**data)
        
        assert stats_response.total_memories >= 0
        assert stats_response.active_sessions >= 0
        assert stats_response.uptime_seconds >= 0
        assert isinstance(stats_response.memory_store_stats, dict)
        assert isinstance(stats_response.api_stats, dict)

    def test_version_endpoint(self, test_client):
        """Test version endpoint."""
        response = test_client.get("/api/v1/version")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "0.8-alpha"
        assert data["api_version"] == "v1"
        assert "python_version" in data
        assert "platform" in data
        assert "architecture" in data

    def test_metrics_endpoint_enabled(self, test_client):
        """Test metrics endpoint when enabled."""
        response = test_client.get("/api/v1/metrics")
        # Should be 200 if metrics enabled or 404 if disabled
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            # Should be Prometheus format
            assert response.headers.get("content-type") == "text/plain; version=0.0.4; charset=utf-8"
            content = response.text
            assert "# HELP" in content or "# TYPE" in content

    def test_metrics_endpoint_disabled(self, test_client, monkeypatch):
        """Test metrics endpoint when disabled."""
        # This would require mocking the settings to disable metrics
        pass


class TestAdminEndpoints:
    """Test admin endpoints."""

    def test_maintenance_mode_status(self, test_client):
        """Test getting maintenance mode status."""
        response = test_client.get("/api/v1/admin/maintenance-mode")
        assert response.status_code in [200, 501]  # 501 if not configured
        
        if response.status_code == 200:
            data = response.json()
            assert "enabled" in data
            assert isinstance(data["enabled"], bool)

    def test_maintenance_mode_enable(self, test_client):
        """Test enabling maintenance mode."""
        response = test_client.post("/api/v1/admin/maintenance-mode/enable")
        assert response.status_code in [204, 501]  # 501 if not configured

    def test_maintenance_mode_disable(self, test_client):
        """Test disabling maintenance mode."""
        response = test_client.post("/api/v1/admin/maintenance-mode/disable")
        assert response.status_code in [204, 501]  # 501 if not configured


class TestSchemas:
    """Test Pydantic schemas."""

    def test_memory_create_schema(self):
        """Test MemoryCreate schema."""
        # Valid data
        data = {
            "text": "Test memory text",
            "role": "user",
            "tags": ["test", "memory"]
        }
        memory_create = MemoryCreate(**data)
        assert memory_create.text == "Test memory text"
        assert memory_create.role == "user"
        assert memory_create.tags == ["test", "memory"]
        assert memory_create.user_id is None

        # With user_id
        data["user_id"] = "user123"
        memory_create = MemoryCreate(**data)
        assert memory_create.user_id == "user123"

    def test_memory_create_schema_validation(self):
        """Test MemoryCreate schema validation."""
        # Empty text should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="")

        # Text too long should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="x" * 10001)

        # Role too long should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="test", role="x" * 33)

        # Too many tags should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="test", tags=["tag"] * 11)

    def test_memory_read_schema(self):
        """Test MemoryRead schema."""
        from datetime import datetime
        
        data = {
            "id": "mem123",
            "user_id": "user123",
            "text": "Test memory text",
            "role": "user",
            "tags": ["test"],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        memory_read = MemoryRead(**data)
        assert memory_read.id == "mem123"
        assert memory_read.user_id == "user123"
        assert memory_read.text == "Test memory text"
        assert memory_read.role == "user"
        assert memory_read.tags == ["test"]
        assert isinstance(memory_read.created_at, datetime)
        assert isinstance(memory_read.updated_at, datetime)

    def test_memory_update_schema(self):
        """Test MemoryUpdate schema."""
        from memory_system.api.schemas import MemoryUpdate
        
        # All fields optional
        update = MemoryUpdate()
        assert update.text is None
        assert update.role is None
        assert update.tags is None

        # Partial update
        update = MemoryUpdate(text="Updated text")
        assert update.text == "Updated text"
        assert update.role is None
        assert update.tags is None

    def test_memory_query_schema(self):
        """Test MemoryQuery schema."""
        query = MemoryQuery(query="test query")
        assert query.query == "test query"
        assert query.top_k == 10
        assert query.include_embeddings is False

        # Custom values
        query = MemoryQuery(
            query="test query",
            top_k=5,
            include_embeddings=True
        )
        assert query.query == "test query"
        assert query.top_k == 5
        assert query.include_embeddings is True

    def test_memory_query_schema_validation(self):
        """Test MemoryQuery schema validation."""
        # Empty query should fail
        with pytest.raises(ValueError):
            MemoryQuery(query="")

        # Query too long should fail
        with pytest.raises(ValueError):
            MemoryQuery(query="x" * 1001)

        # top_k out of range should fail
        with pytest.raises(ValueError):
            MemoryQuery(query="test", top_k=0)

        with pytest.raises(ValueError):
            MemoryQuery(query="test", top_k=101)

    def test_memory_search_result_schema(self):
        """Test MemorySearchResult schema."""
        from datetime import datetime
        
        data = {
            "id": "mem123",
            "user_id": "user123",
            "text": "Test memory text",
            "role": "user",
            "tags": ["test"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "score": 0.95,
            "embedding": [0.1, 0.2, 0.3]
        }
        result = MemorySearchResult(**data)
        assert result.id == "mem123"
        assert result.score == 0.95
        assert result.embedding == [0.1, 0.2, 0.3]

    def test_health_response_schema(self):
        """Test HealthResponse schema."""
        data = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "uptime_seconds": 3600,
            "version": "0.8-alpha",
            "checks": {"database": True, "index": True},
            "memory_store_health": {"total_memories": 100},
            "api_enabled": True
        }
        health = HealthResponse(**data)
        assert health.status == "healthy"
        assert health.timestamp == "2024-01-01T00:00:00Z"
        assert health.uptime_seconds == 3600
        assert health.version == "0.8-alpha"
        assert health.checks == {"database": True, "index": True}
        assert health.memory_store_health == {"total_memories": 100}
        assert health.api_enabled is True

    def test_stats_response_schema(self):
        """Test StatsResponse schema."""
        data = {
            "total_memories": 1000,
            "active_sessions": 50,
            "uptime_seconds": 3600,
            "memory_store_stats": {"cache_hit_rate": 0.85},
            "api_stats": {"requests_per_second": 100}
        }
        stats = StatsResponse(**data)
        assert stats.total_memories == 1000
        assert stats.active_sessions == 50
        assert stats.uptime_seconds == 3600
        assert stats.memory_store_stats == {"cache_hit_rate": 0.85}
        assert stats.api_stats == {"requests_per_second": 100}

    def test_success_response_schema(self):
        """Test SuccessResponse schema."""
        success = SuccessResponse()
        assert success.message == "success"
        assert success.api_version == "v1"

        success = SuccessResponse(message="custom success")
        assert success.message == "custom success"
        assert success.api_version == "v1"

    def test_error_response_schema(self):
        """Test ErrorResponse schema."""
        error = ErrorResponse(detail="Something went wrong")
        assert error.detail == "Something went wrong"
        assert error.api_version == "v1"


class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.get("/api/v1/health")
        # CORS headers should be present if enabled
        # This depends on the test configuration

    def test_rate_limiting_middleware(self, test_client):
        """Test rate limiting middleware."""
        # Make multiple requests to trigger rate limiting
        responses = []
        for i in range(10):
            response = test_client.get("/api/v1/health")
            responses.append(response)
        
        # All should succeed for a small number of requests
        for response in responses:
            assert response.status_code in [200, 429]

    def test_maintenance_mode_middleware(self, test_client):
        """Test maintenance mode middleware."""
        # This would require enabling maintenance mode
        # and testing that non-exempt endpoints return 503
        pass


class TestAsyncEndpoints:
    """Test async endpoints."""

    async def test_async_health_endpoint(self, async_test_client):
        """Test health endpoint with async client."""
        response = await async_test_client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "0.8-alpha"
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    async def test_async_stats_endpoint(self, async_test_client):
        """Test stats endpoint with async client."""
        response = await async_test_client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_memories" in data
        assert "active_sessions" in data
        assert "uptime_seconds" in data

    async def test_async_version_endpoint(self, async_test_client):
        """Test version endpoint with async client."""
        response = await async_test_client.get("/api/v1/version")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "0.8-alpha"
        assert data["api_version"] == "v1"


class TestErrorHandling:
    """Test error handling."""

    def test_404_error(self, test_client):
        """Test 404 error handling."""
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_405_error(self, test_client):
        """Test 405 error handling."""
        response = test_client.post("/api/v1/health")
        assert response.status_code == 405

    def test_422_validation_error(self, test_client):
        """Test 422 validation error handling."""
        # This would require an endpoint that accepts POST data
        # and test with invalid data
        pass


class TestApplicationLifecycle:
    """Test application lifecycle events."""

    def test_app_startup(self, test_app):
        """Test application startup."""
        assert isinstance(test_app, FastAPI)
        assert test_app.title == "Unified Memory System"
        assert test_app.version == "0.8-alpha"

    def test_app_shutdown(self, test_app):
        """Test application shutdown."""
        # This would require testing the lifespan context manager
        pass


class TestDependencyInjection:
    """Test dependency injection."""

    def test_settings_dependency(self, test_client):
        """Test settings dependency is properly injected."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        # The fact that we get a successful response means
        # the settings dependency was properly injected

    def test_memory_store_dependency(self, test_client):
        """Test memory store dependency is properly injected."""
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200
        # The fact that we get a successful response means
        # the memory store dependency was properly injected


class TestConcurrency:
    """Test concurrent access."""

    async def test_concurrent_health_checks(self, async_test_client):
        """Test concurrent health check requests."""
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                async_test_client.get("/api/v1/health")
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    async def test_concurrent_stats_requests(self, async_test_client):
        """Test concurrent stats requests."""
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                async_test_client.get("/api/v1/stats")
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestPerformance:
    """Test performance characteristics."""

    def test_health_endpoint_performance(self, test_client):
        """Test health endpoint performance."""
        start_time = time.time()
        
        for i in range(100):
            response = test_client.get("/api/v1/health")
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 100 requests in reasonable time
        assert total_time < 10.0  # 10 seconds max
        
        # Average response time should be reasonable
        avg_time = total_time / 100
        assert avg_time < 0.1  # 100ms max per request

    def test_stats_endpoint_performance(self, test_client):
        """Test stats endpoint performance."""
        start_time = time.time()
        
        for i in range(50):
            response = test_client.get("/api/v1/stats")
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 50 requests in reasonable time
        assert total_time < 10.0  # 10 seconds max
