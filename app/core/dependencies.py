from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.core.config import SECRET_KEY, ALGORITHM

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail={ "success": False, "message": "Invalid token type"})

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail={"success": False, "message": "Invalid token payload"})

        return user_id

    except JWTError:
        print(JWTError)
        raise HTTPException(status_code=401, detail={"success": False, "message": "Invalid or expired token"})

