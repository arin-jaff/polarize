from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ActivitySummary(BaseModel):
    id: str
    sport: str
    sub_sport: Optional[str] = None
    name: Optional[str] = None
    start_time: datetime
    total_timer_time: float
    total_distance: Optional[float] = None
    avg_heart_rate: Optional[int] = None
    avg_power: Optional[int] = None
    normalized_power: Optional[float] = None
    tss: Optional[float] = None
    scaled_tss: Optional[float] = None
    source: str


class ActivityDetail(ActivitySummary):
    end_time: Optional[datetime] = None
    total_elapsed_time: Optional[float] = None
    total_calories: Optional[int] = None
    max_heart_rate: Optional[int] = None
    max_power: Optional[int] = None
    avg_cadence: Optional[int] = None
    avg_speed: Optional[float] = None
    max_speed: Optional[float] = None
    total_ascent: Optional[float] = None
    total_descent: Optional[float] = None
    avg_stroke_rate: Optional[int] = None
    intensity_factor: Optional[float] = None
    description: Optional[str] = None
    is_combined: bool = False
    has_records: bool = False


class DuplicateCandidate(BaseModel):
    existing_id: str
    existing_name: Optional[str] = None
    existing_start: datetime
    existing_end: Optional[datetime] = None
    new_start: datetime
    new_end: Optional[datetime] = None
    overlap_seconds: float


class CombineRequest(BaseModel):
    activity_id_1: str
    activity_id_2: str
    time_offset_ms: int = 0  # manual alignment offset
    prefer_data_from: int = 1  # 1 or 2: which file's HR/power data to prefer on conflict
