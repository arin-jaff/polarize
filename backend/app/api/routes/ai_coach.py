from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json

from app.core.auth import get_current_user
from app.core.config import settings
from app.models.user import User

router = APIRouter()

SYSTEM_PROMPT = (
    "You are an expert endurance sports coach specializing in rowing, cycling, "
    "running, and triathlon. You provide evidence-based, personalized training advice. "
    "You consider the athlete's current fitness level, goals, available time, and "
    "recovery status when making recommendations. Always prioritize injury prevention "
    "and sustainable training load."
)


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    response: str


def _build_messages(user: User, request: ChatRequest) -> list[dict]:
    """Build message list with athlete context."""
    system = SYSTEM_PROMPT
    system += (
        f"\n\nAthlete Profile: "
        f"Name={user.name}, "
        f"Primary Sport={user.primary_sport}, "
        f"Current Fitness (CTL)={user.current_ctl:.0f}, "
        f"Current Fatigue (ATL)={user.current_atl:.0f}, "
        f"Current Form (TSB)={user.current_ctl - user.current_atl:.0f}"
    )
    if user.thresholds.threshold_hr:
        system += f", LTHR={user.thresholds.threshold_hr} bpm"
    if user.thresholds.threshold_power:
        system += f", FTP={user.thresholds.threshold_power}W"

    messages = [{"role": "system", "content": system}]
    messages.extend(request.conversation_history)
    messages.append({"role": "user", "content": request.message})
    return messages


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    """Send a message to the AI coach and get a response."""
    messages = _build_messages(user, request)
    payload = {
        "model": settings.ollama_model_name,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return ChatResponse(response=data["message"]["content"])
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="AI coach is not available. Ensure Ollama is running.",
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"AI coach error: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, user: User = Depends(get_current_user)):
    """Stream a response from the AI coach."""
    messages = _build_messages(user, request)
    payload = {
        "model": settings.ollama_model_name,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
    }

    async def generate():
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_base_url}/api/chat",
                    json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if not data.get("done", False):
                                yield f"data: {json.dumps({'text': data['message']['content']})}\n\n"
                            else:
                                yield "data: [DONE]\n\n"
            except httpx.ConnectError:
                yield f"data: {json.dumps({'error': 'AI coach is not available'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
