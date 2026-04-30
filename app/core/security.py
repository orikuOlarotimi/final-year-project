from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from app.core.config import SECRET_KEY, ALGORITHM

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 🔹 Hash password
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# 🔹 Verify password
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# 🔹 Create tokens
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()

    expire = datetime.utcnow() + expires_delta

    to_encode.update({
        "exp": expire,
        "type": "access",
        "sub": data["sub"]
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()

    expire = datetime.utcnow() + expires_delta

    to_encode.update({
        "exp": expire,
        "type": "refresh",
        "sub": data["sub"]
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
