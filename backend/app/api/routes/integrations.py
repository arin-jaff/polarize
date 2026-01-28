from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import get_current_user
from app.core.config import settings
from app.models.user import User

router = APIRouter()


# ---------- Garmin (OAuth 1.0a) ----------

@router.get("/garmin/connect")
async def garmin_connect(user: User = Depends(get_current_user)):
    """Initiate Garmin OAuth 1.0a flow."""
    if not settings.garmin_consumer_key:
        raise HTTPException(status_code=501, detail="Garmin integration not configured")
    # TODO: Implement OAuth 1.0a request token flow
    return {"message": "Garmin OAuth not yet implemented", "status": "pending"}


@router.get("/garmin/callback")
async def garmin_callback(request: Request):
    """Handle Garmin OAuth callback."""
    # TODO: Exchange request token for access token
    return {"message": "Garmin callback not yet implemented"}


@router.post("/garmin/sync")
async def garmin_sync(user: User = Depends(get_current_user)):
    """Manually trigger sync of recent Garmin activities."""
    if not user.garmin.connected:
        raise HTTPException(status_code=400, detail="Garmin not connected")
    # TODO: Pull recent activities via Garmin Health API
    return {"message": "Garmin sync not yet implemented"}


@router.delete("/garmin/disconnect")
async def garmin_disconnect(user: User = Depends(get_current_user)):
    """Disconnect Garmin integration."""
    user.garmin.connected = False
    user.garmin.access_token = None
    user.garmin.access_token_secret = None
    await user.save()
    return {"status": "disconnected"}


# ---------- Concept2 (OAuth 2.0) ----------

@router.get("/concept2/connect")
async def concept2_connect(user: User = Depends(get_current_user)):
    """Initiate Concept2 OAuth 2.0 flow."""
    if not settings.concept2_client_id:
        raise HTTPException(status_code=501, detail="Concept2 integration not configured")

    auth_url = (
        f"https://log.concept2.com/oauth/authorize"
        f"?client_id={settings.concept2_client_id}"
        f"&redirect_uri={settings.concept2_redirect_uri}"
        f"&response_type=code"
        f"&scope=user:read,results:read"
    )
    return {"auth_url": auth_url}


@router.get("/concept2/callback")
async def concept2_callback(code: str, request: Request):
    """Handle Concept2 OAuth callback and exchange code for token."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://log.concept2.com/oauth/access_token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": settings.concept2_client_id,
                "client_secret": settings.concept2_client_secret,
                "redirect_uri": settings.concept2_redirect_uri,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange Concept2 token")

        token_data = resp.json()

    # TODO: Associate token with user (need user context from state param)
    return {"message": "Concept2 connected", "token_data": token_data}


@router.post("/concept2/sync")
async def concept2_sync(user: User = Depends(get_current_user)):
    """Pull recent workouts from Concept2 Logbook."""
    if not user.concept2.connected:
        raise HTTPException(status_code=400, detail="Concept2 not connected")

    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://log.concept2.com/api/users/me/results",
            headers={"Authorization": f"Bearer {user.concept2.access_token}"},
            params={"type": "rower"},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch Concept2 data")

        results = resp.json()

    # TODO: Convert Concept2 results to Activity documents
    return {"synced_count": len(results.get("data", [])), "results": results}


@router.delete("/concept2/disconnect")
async def concept2_disconnect(user: User = Depends(get_current_user)):
    """Disconnect Concept2 integration."""
    user.concept2.connected = False
    user.concept2.access_token = None
    user.concept2.refresh_token = None
    await user.save()
    return {"status": "disconnected"}
