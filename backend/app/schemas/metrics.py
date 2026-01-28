from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class DailyMetrics(BaseModel):
    date: date
    tss: float = 0.0
    scaled_tss: float = 0.0
    ctl: float = 0.0  # Fitness
    atl: float = 0.0  # Fatigue
    tsb: float = 0.0  # Form


class MetricsRange(BaseModel):
    start_date: date
    end_date: date
    daily: list[DailyMetrics]
    current_ctl: float
    current_atl: float
    current_tsb: float


class WeeklySummary(BaseModel):
    week_start: date
    total_tss: float
    total_scaled_tss: float
    total_duration: float  # seconds
    total_distance: float  # meters
    activity_count: int
    by_sport: dict[str, float]  # sport -> TSS


class PerformanceSnapshot(BaseModel):
    """Current performance metrics dashboard."""
    fitness: float  # CTL
    fatigue: float  # ATL
    form: float  # TSB
    total_tss_7d: float
    total_tss_28d: float
    total_duration_7d: float
    total_distance_7d: float
    ramp_rate_7d: float  # CTL change per week
    ramp_rate_28d: float
    ramp_rate_90d: float
