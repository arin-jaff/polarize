from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.core.config import settings

client: AsyncIOMotorClient = None


async def init_db():
    """Initialize MongoDB connection and Beanie ODM."""
    global client
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db_name]

    # Import all document models here
    from app.models.user import User
    from app.models.activity import Activity
    from app.models.workout import PlannedWorkout

    await init_beanie(
        database=db,
        document_models=[User, Activity, PlannedWorkout],
    )


async def close_db():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
