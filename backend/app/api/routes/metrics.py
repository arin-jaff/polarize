from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query

from app.core.auth import get_current_user
from app.models.user import User
from app.schemas.metrics import MetricsRange, PerformanceSnapshot, WeeklySummary
from app.services.metrics import (
    compute_metrics_range,
    get_performance_snapshot,
    get_weekly_summaries,
)

router = APIRouter()


@router.get("/range", response_model=MetricsRange)
async def get_metrics_range(
    start: date = Query(...),
    end: date = Query(...),
    user: User = Depends(get_current_user),
):
    """Get daily CTL/ATL/TSB for a date range."""
    return await compute_metrics_range(str(user.id), start, end, user)


@router.get("/snapshot", response_model=PerformanceSnapshot)
async def get_snapshot(user: User = Depends(get_current_user)):
    """Get current performance snapshot (fitness, fatigue, form)."""
    return await get_performance_snapshot(str(user.id), user)


@router.get("/weekly", response_model=list[WeeklySummary])
async def get_weekly(
    weeks: int = Query(12, le=52),
    user: User = Depends(get_current_user),
):
    """Get weekly training summaries."""
    return await get_weekly_summaries(str(user.id), weeks)
