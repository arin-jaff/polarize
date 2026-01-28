from datetime import datetime, timezone
from typing import Optional

from beanie import Document
from pydantic import BaseModel, EmailStr, Field


class SportScaling(BaseModel):
    """Scaling factor for a sport relative to the user's primary sport."""
    sport: str
    scaling_factor: float = 1.0  # e.g., 0.5 means 50% TSS effectiveness


class ThresholdValues(BaseModel):
    """User's physiological threshold values."""
    # Heart rate
    threshold_hr: Optional[int] = None  # LTHR in bpm
    max_hr: Optional[int] = None
    resting_hr: Optional[int] = None

    # Power
    threshold_power: Optional[int] = None  # FTP in watts
    running_threshold_power: Optional[int] = None  # rFTP in watts
    critical_power: Optional[int] = None  # Stryd CP in watts

    # Pace
    threshold_pace: Optional[float] = None  # seconds per km


class ZoneConfig(BaseModel):
    """User's chosen zone calculation method."""
    hr_method: str = "joe_friel"  # Zone calculation method ID
    hr_activity: str = "running"  # cycling or running (some methods differ)
    power_method: str = "andy_coggan"
    power_activity: str = "cycling"


class GarminIntegration(BaseModel):
    """Garmin OAuth 1.0a tokens."""
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None
    connected: bool = False


class Concept2Integration(BaseModel):
    """Concept2 OAuth 2.0 tokens."""
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    connected: bool = False


class CoachSettings(BaseModel):
    """AI Coach configuration."""
    # Coach type: specialist, generalist, recreational
    coach_type: str = "specialist"

    # Training plan type: polarized, traditional, threshold
    training_plan: str = "polarized"

    # Time constraint: minimal (0-5h), moderate (5-10h), committed (10-15h),
    #                  serious (15-20h), elite (20+h)
    time_constraint: str = "moderate"

    # Weekly hours available (specific number for fine-tuning)
    weekly_hours_available: Optional[float] = None


class User(Document):
    email: EmailStr
    hashed_password: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Sport settings
    primary_sport: str = "rowing"  # rowing, cycling, running, triathlon, etc.
    sport_scaling: list[SportScaling] = Field(default_factory=list)

    # Thresholds
    thresholds: ThresholdValues = Field(default_factory=ThresholdValues)
    zone_config: ZoneConfig = Field(default_factory=ZoneConfig)

    # Integrations
    garmin: GarminIntegration = Field(default_factory=GarminIntegration)
    concept2: Concept2Integration = Field(default_factory=Concept2Integration)

    # AI Coach settings
    coach_settings: CoachSettings = Field(default_factory=CoachSettings)

    # Metrics history (CTL/ATL are rolling, we store the latest)
    current_ctl: float = 0.0
    current_atl: float = 0.0

    class Settings:
        name = "users"
