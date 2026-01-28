from pydantic import BaseModel


class Zone(BaseModel):
    zone_number: int
    name: str
    lower: float  # absolute value (bpm or watts)
    upper: float


class ZoneResult(BaseModel):
    method: str
    activity: str  # cycling, running, etc.
    threshold_type: str  # LTHR, FTP, CP, etc.
    threshold_value: float
    zones: list[Zone]


class ZoneMethodInfo(BaseModel):
    method_id: str
    name: str
    zone_count: int
    threshold_type: str  # LTHR, FTP, CP, CTS_HR, CTS_Power
    supports: list[str]  # cycling, running, rowing, etc.


class UpdateThresholds(BaseModel):
    threshold_hr: int | None = None
    max_hr: int | None = None
    resting_hr: int | None = None
    threshold_power: int | None = None
    running_threshold_power: int | None = None
    critical_power: int | None = None


class UpdateZoneConfig(BaseModel):
    hr_method: str | None = None
    hr_activity: str | None = None
    power_method: str | None = None
    power_activity: str | None = None
