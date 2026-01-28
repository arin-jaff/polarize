from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from app.core.auth import get_current_user
from app.models.user import User
from app.models.activity import Activity
from app.schemas.activity import ActivitySummary, ActivityDetail, DuplicateCandidate, CombineRequest
from app.services.fit_parser import parse_fit_file
from app.services.metrics import compute_activity_metrics
from app.services.duplicate_detector import find_duplicates
from app.services.fit_combiner import combine_activities

router = APIRouter()


@router.post("/upload", response_model=ActivityDetail | dict)
async def upload_fit_file(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Upload a .FIT file. Returns activity detail or duplicate candidates."""
    if not file.filename.lower().endswith(".fit"):
        raise HTTPException(status_code=400, detail="Only .FIT files are supported")

    contents = await file.read()
    activity = await parse_fit_file(contents, str(user.id), file.filename)

    # Check for duplicates before saving
    duplicates = await find_duplicates(activity, str(user.id))
    if duplicates:
        # Save as pending, return duplicate info
        await activity.insert()
        return {
            "activity": _to_detail(activity),
            "duplicates": duplicates,
            "message": "Potential duplicate activities found. Would you like to combine?",
        }

    # Compute metrics and save
    await compute_activity_metrics(activity, user)
    await activity.insert()
    return _to_detail(activity)


@router.post("/combine", response_model=ActivityDetail)
async def combine_fit_files(
    req: CombineRequest,
    user: User = Depends(get_current_user),
):
    """Combine two overlapping activities into one."""
    combined = await combine_activities(
        req.activity_id_1,
        req.activity_id_2,
        str(user.id),
        req.time_offset_ms,
        req.prefer_data_from,
    )
    await compute_activity_metrics(combined, user)
    await combined.insert()
    return _to_detail(combined)


@router.get("/", response_model=list[ActivitySummary])
async def list_activities(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    sport: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: User = Depends(get_current_user),
):
    """List activities with optional date range and sport filters."""
    query = Activity.find(Activity.user_id == str(user.id))

    if start:
        query = query.find(Activity.start_time >= start)
    if end:
        query = query.find(Activity.start_time <= end)
    if sport:
        query = query.find(Activity.sport == sport)

    activities = await query.sort(-Activity.start_time).skip(offset).limit(limit).to_list()
    return [_to_summary(a) for a in activities]


@router.get("/{activity_id}", response_model=ActivityDetail)
async def get_activity(activity_id: str, user: User = Depends(get_current_user)):
    activity = await Activity.get(activity_id)
    if not activity or activity.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Activity not found")
    return _to_detail(activity)


@router.get("/{activity_id}/records")
async def get_activity_records(activity_id: str, user: User = Depends(get_current_user)):
    """Get the time-series record data for graphing."""
    activity = await Activity.get(activity_id)
    if not activity or activity.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Activity not found")
    return {"records": [r.model_dump() for r in activity.records]}


@router.delete("/{activity_id}", status_code=204)
async def delete_activity(activity_id: str, user: User = Depends(get_current_user)):
    activity = await Activity.get(activity_id)
    if not activity or activity.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Activity not found")
    await activity.delete()


def _to_summary(a: Activity) -> ActivitySummary:
    return ActivitySummary(
        id=str(a.id),
        sport=a.sport,
        sub_sport=a.sub_sport,
        name=a.name,
        start_time=a.start_time,
        total_timer_time=a.total_timer_time,
        total_distance=a.total_distance,
        avg_heart_rate=a.avg_heart_rate,
        avg_power=a.avg_power,
        normalized_power=a.normalized_power,
        tss=a.tss,
        scaled_tss=a.scaled_tss,
        source=a.source,
    )


def _to_detail(a: Activity) -> ActivityDetail:
    return ActivityDetail(
        id=str(a.id),
        sport=a.sport,
        sub_sport=a.sub_sport,
        name=a.name,
        start_time=a.start_time,
        total_timer_time=a.total_timer_time,
        total_distance=a.total_distance,
        avg_heart_rate=a.avg_heart_rate,
        avg_power=a.avg_power,
        normalized_power=a.normalized_power,
        tss=a.tss,
        scaled_tss=a.scaled_tss,
        source=a.source,
        end_time=a.end_time,
        total_elapsed_time=a.total_elapsed_time,
        total_calories=a.total_calories,
        max_heart_rate=a.max_heart_rate,
        max_power=a.max_power,
        avg_cadence=a.avg_cadence,
        avg_speed=a.avg_speed,
        max_speed=a.max_speed,
        total_ascent=a.total_ascent,
        total_descent=a.total_descent,
        avg_stroke_rate=a.avg_stroke_rate,
        intensity_factor=a.intensity_factor,
        description=a.description,
        is_combined=a.is_combined,
        has_records=len(a.records) > 0,
    )
