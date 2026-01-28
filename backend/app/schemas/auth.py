from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    primary_sport: str = "rowing"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    primary_sport: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
