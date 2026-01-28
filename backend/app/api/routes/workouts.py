from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import get_current_user
from app.models.user import User
from app.models.workout import PlannedWorkout, WorkoutStep

router = APIRouter()


class WorkoutCreate(BaseModel):
    scheduled_date: datetime
    name: str
    description: Optional[str] = None
    sport: str = "other"
    estimated_duration: Optional[float] = None
    estimated_tss: Optional[float] = None
    steps: list[WorkoutStep] = []
    pre_activity_comments: Optional[str] = None


class WorkoutUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[datetime] = None
    sport: Optional[str] = None
    estimated_duration: Optional[float] = None
    estimated_tss: Optional[float] = None
    steps: Optional[list[WorkoutStep]] = None
    pre_activity_comments: Optional[str] = None
    post_activity_comments: Optional[str] = None
    completed: Optional[bool] = None
    activity_id: Optional[str] = None


@router.post("/", status_code=201)
async def create_workout(
    data: WorkoutCreate,
    user: User = Depends(get_current_user),
):
    workout = PlannedWorkout(
        user_id=str(user.id),
        **data.model_dump(),
    )
    await workout.insert()
    return _to_response(workout)


@router.get("/")
async def list_workouts(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    user: User = Depends(get_current_user),
):
    query = PlannedWorkout.find(PlannedWorkout.user_id == str(user.id))
    if start:
        query = query.find(PlannedWorkout.scheduled_date >= start)
    if end:
        query = query.find(PlannedWorkout.scheduled_date <= end)

    workouts = await query.sort(PlannedWorkout.scheduled_date).to_list()
    return [_to_response(w) for w in workouts]


@router.get("/{workout_id}")
async def get_workout(workout_id: str, user: User = Depends(get_current_user)):
    workout = await PlannedWorkout.get(workout_id)
    if not workout or workout.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Workout not found")
    return _to_response(workout)


@router.put("/{workout_id}")
async def update_workout(
    workout_id: str,
    data: WorkoutUpdate,
    user: User = Depends(get_current_user),
):
    workout = await PlannedWorkout.get(workout_id)
    if not workout or workout.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Workout not found")

    update_data = data.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(workout, key, value)
    await workout.save()
    return _to_response(workout)


@router.delete("/{workout_id}", status_code=204)
async def delete_workout(workout_id: str, user: User = Depends(get_current_user)):
    workout = await PlannedWorkout.get(workout_id)
    if not workout or workout.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Workout not found")
    await workout.delete()


def _to_response(w: PlannedWorkout) -> dict:
    return {
        "id": str(w.id),
        "user_id": w.user_id,
        "scheduled_date": w.scheduled_date.isoformat(),
        "completed": w.completed,
        "activity_id": w.activity_id,
        "name": w.name,
        "description": w.description,
        "sport": w.sport,
        "estimated_duration": w.estimated_duration,
        "estimated_tss": w.estimated_tss,
        "steps": [s.model_dump() for s in w.steps],
        "pre_activity_comments": w.pre_activity_comments,
        "post_activity_comments": w.post_activity_comments,
    }
