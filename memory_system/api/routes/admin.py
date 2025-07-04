"""Admin routes exposed under `/api/v1/admin`."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from memory_system.api.middleware import MaintenanceModeMiddleware
from memory_system.api.app import get_maintenance_middleware_instance

router = APIRouter(prefix="/admin", tags=["Administration"])

def _maintenance() -> MaintenanceModeMiddleware:
    """Internal dependency to get the maintenance mode middleware (or raise error)."""
    mw = get_maintenance_middleware_instance()
    if mw is None:
        # 501 Not Implemented if maintenance middleware is not configured
        raise HTTPException(status_code=501, detail="Maintenance middleware not configured")
    return mw

@router.get("/maintenance-mode", summary="Get maintenance mode state", response_model=dict)
async def maintenance_status(mw: MaintenanceModeMiddleware = Depends(_maintenance)) -> Dict[str, bool]:
    """Check whether maintenance mode is currently enabled."""
    return {"enabled": mw._enabled}

@router.post("/maintenance-mode/enable", status_code=status.HTTP_204_NO_CONTENT, summary="Enable maintenance mode")
async def enable_maintenance(mw: MaintenanceModeMiddleware = Depends(_maintenance)) -> None:
    """Switch maintenance mode **on** (returns 204 No Content on success)."""
    mw.enable()

@router.post("/maintenance-mode/disable", status_code=status.HTTP_204_NO_CONTENT, summary="Disable maintenance mode")
async def disable_maintenance(mw: MaintenanceModeMiddleware = Depends(_maintenance)) -> None:
    """Switch maintenance mode **off** and restore normal operation."""
    mw.disable()
