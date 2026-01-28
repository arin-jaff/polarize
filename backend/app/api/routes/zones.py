from fastapi import APIRouter, Depends

from app.core.auth import get_current_user
from app.models.user import User
from app.schemas.zones import ZoneResult, ZoneMethodInfo, UpdateThresholds, UpdateZoneConfig
from app.services.zone_calculator import (
    calculate_hr_zones,
    calculate_power_zones,
    get_available_hr_methods,
    get_available_power_methods,
)

router = APIRouter()


@router.get("/hr/methods", response_model=list[ZoneMethodInfo])
async def list_hr_methods():
    """List all available heart rate zone calculation methods."""
    return get_available_hr_methods()


@router.get("/power/methods", response_model=list[ZoneMethodInfo])
async def list_power_methods():
    """List all available power zone calculation methods."""
    return get_available_power_methods()


@router.get("/hr", response_model=ZoneResult)
async def get_hr_zones(user: User = Depends(get_current_user)):
    """Get the user's current heart rate zones."""
    return calculate_hr_zones(
        method=user.zone_config.hr_method,
        activity=user.zone_config.hr_activity,
        threshold_hr=user.thresholds.threshold_hr,
        max_hr=user.thresholds.max_hr,
        resting_hr=user.thresholds.resting_hr,
    )


@router.get("/power", response_model=ZoneResult)
async def get_power_zones(user: User = Depends(get_current_user)):
    """Get the user's current power zones."""
    return calculate_power_zones(
        method=user.zone_config.power_method,
        activity=user.zone_config.power_activity,
        threshold_power=user.thresholds.threshold_power,
        running_threshold_power=user.thresholds.running_threshold_power,
        critical_power=user.thresholds.critical_power,
    )


@router.put("/thresholds")
async def update_thresholds(
    data: UpdateThresholds,
    user: User = Depends(get_current_user),
):
    """Update the user's threshold values (LTHR, FTP, CP, etc.)."""
    update_data = data.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(user.thresholds, key, value)
    await user.save()
    return {"status": "updated"}


@router.put("/config")
async def update_zone_config(
    data: UpdateZoneConfig,
    user: User = Depends(get_current_user),
):
    """Update the user's zone calculation method preferences."""
    update_data = data.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(user.zone_config, key, value)
    await user.save()
    return {"status": "updated"}
