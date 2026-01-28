from fastapi import APIRouter, HTTPException, status

from app.core.auth import get_password_hash, verify_password, create_access_token
from app.models.user import User
from app.schemas.auth import UserCreate, UserLogin, UserResponse, Token

router = APIRouter()


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(data: UserCreate):
    existing = await User.find_one(User.email == data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=data.email,
        hashed_password=get_password_hash(data.password),
        name=data.name,
        primary_sport=data.primary_sport,
    )
    await user.insert()
    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@router.post("/login", response_model=Token)
async def login(data: UserLogin):
    user = await User.find_one(User.email == data.email)
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = __import__("fastapi").Depends(__import__("app.core.auth", fromlist=["get_current_user"]).get_current_user)):
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        primary_sport=user.primary_sport,
    )
