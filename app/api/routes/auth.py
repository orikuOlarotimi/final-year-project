from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
import random

from app.models.user import User
from app.models.otp import OTP
from app.models.token import RefreshToken
from app.core.security import hash_password, create_access_token, create_refresh_token, verify_password
from app.schemas.schemas import RegisterSchema
from app.services.email_service import EmailService
from app.schemas.schemas import VerifyOTPSchema, LoginSchema, RefreshTokenSchema, ResendOTPSchema, ForgotPasswordSchema, ResetPasswordSchema
from app.core.dependencies import get_current_user
from jose import jwt, JWTError, ExpiredSignatureError
from app.core.config import SECRET_KEY, ALGORITHM


router = APIRouter(prefix="/auth", tags=["Auth"])

email_service = EmailService()


@router.post("/signup")
async def signup(payload: RegisterSchema):
    try:
        # 1. Normalize email
        if not payload.email or payload.email.strip() == "":
            raise HTTPException(status_code=400, detail={"success": False,
                                                         "message": "Email is required"})
        email = payload.email.strip().lower()

        # 1. Check if user exists
        user = await User.find_one(User.email == email)

        if user and user.is_verified:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "status": "verified",
                    "action": "LOGIN",
                    "message": "User already exists. Please login"
                }
            )

        if user and not user.is_verified:
            otp_code = str(random.randint(100000, 999999))
            # remove old OTPs (optional but clean)
            await OTP.find(OTP.email == email).delete()

            otp = OTP(
                email=email,
                code=otp_code,
                expires_at=datetime.utcnow() + timedelta(minutes=10)
                # is_used=False
            )
            await otp.insert()
            email_service.send_otp_email(email, otp_code)
            return {
                "success": True,
                "status": "pending",
                "action": "VERIFY_OTP",
                "message": "OTP resent. Please verify your account"
            }

        hashed_password = hash_password(payload.password)

        new_user = User(
            email=email,
            password=hashed_password,
            name=payload.name.strip(),
            is_verified=False,
            is_active=True
        )

        await new_user.insert()

        otp_code = str(random.randint(100000, 999999))

        otp = OTP(
            email=email,
            code=otp_code,
            expires_at=datetime.utcnow() + timedelta(minutes=10),
            # is_used=False
        )

        await otp.insert()

        email_service.send_otp_email(email, otp_code)

        return {
            "success": True,
            "status": "pending",
            "action": "VERIFY_OTP",
            "message": "User created. OTP sent to email"
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "status": "error",
                "message": "Signup failed unexpectedly"
            }
        )


@router.post("/verify-otp")
async def verify_otp(payload: VerifyOTPSchema):

    try:
        email = payload.email.strip().lower()
        otp_input = payload.otp.strip()

        # 1. Find OTP record
        otp_record = await OTP.find_one(
            OTP.email == email,
            OTP.code == otp_input
        )

        if not otp_record:
            raise HTTPException(status_code=400, detail={"success": False, "message": "Invalid OTP"})
        #
        # if otp_record.is_used:
        #     raise HTTPException(status_code=400, detail="OTP already used")

        # 3. Check expiry
        if otp_record.expires_at < datetime.utcnow():
            raise HTTPException(status_code=400, detail={"success": False, "message": "OTP expired"})

        # 4. Mark OTP as used
        # otp_record.is_used = True
        #
        # await otp_record.save()

        user = await User.find_one(User.email == email)

        if not user:
            raise HTTPException(status_code=404, detail={"success": False,"message": "User not found"})
        if user.is_verified:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "status": "verified",
                    "action": "LOGIN",
                    "message": "User already verified. Please login."
                }
            )
        user.is_verified = True
        user.updated_at = datetime.utcnow()
        await user.save()

        access_token = create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(minutes=15)
        )

        refresh_token = create_refresh_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(days=7)
        )

        hashed_refresh = hash_password(refresh_token)
        refresh_doc = RefreshToken(
            user_id=str(user.id),
            token_hash=hashed_refresh,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        await OTP.find(OTP.email == email).delete()

        await refresh_doc.insert()

        return {
            "success": True,
            "status": "verified",
            "action": "ACCESS_GRANTED",
            "message": "Account verified successfully",
            "access_token": access_token,
            "refresh_token": refresh_token
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "OTP verification failed"}
        )


@router.post("/login")
async def login(payload: LoginSchema):

    try:
        email = payload.email.strip()
        password = payload.password.strip()

        user = await User.find_one(User.email == email)

        if not user:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "User not found. Please signup."}
            )

        if not user.is_verified:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "status": "pending",
                    "action": "VERIFY_OTP",
                    "message": "Account not verified. Please verify your email by going to signup."
                }
            )

        if not verify_password(password, user.password):
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "Invalid credentials"}
            )

        access_token = create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(minutes=15)
        )

        refresh_token = create_refresh_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(days=7)
        )

        hashed_refresh = hash_password(refresh_token)

        refresh_doc = RefreshToken(
            user_id=str(user.id),
            token_hash=hashed_refresh,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )

        await refresh_doc.insert()

        return {
            "success": True,
            "status": "authenticated",
            "message": "Login successful",
            "access_token": access_token,
            "refresh_token": refresh_token
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Login failed"}
        )


@router.post("/logout")
async def logout(user_id: str = Depends(get_current_user)):

    try:
        user = await User.get(user_id)

        if not user:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "User not found"}
            )

        await RefreshToken.find(RefreshToken.user_id == user_id).delete()
        return {
            "success": True,
            "message": "Logged out successfully"
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Logout failed"}
        )


@router.post("/refresh")
async def refresh_token(payload: RefreshTokenSchema):

    try:
        token = payload.refresh_token

        try:
            payload_data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        except ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Refresh token expired"}
            )

        except JWTError:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Invalid refresh token"}
            )

        # =========================
        # 2. Validate token type
        # =========================
        if payload_data.get("type") != "refresh":
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Invalid token type"}
            )

        user_id = payload_data.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Invalid token payload"}
            )


        tokens = await RefreshToken.find(
            RefreshToken.user_id == user_id
        ).to_list()

        if not tokens:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Session not found"}
            )


        matched_token = None

        for t in tokens:
            if verify_password(token, t.token_hash):
                matched_token = t
                break

        if not matched_token:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Invalid refresh token"}
            )

        # =========================
        # 5. Check DB expiry
        # =========================
        if matched_token.expires_at < datetime.utcnow():
            await matched_token.delete()

            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Refresh token expired"}
            )

        # =========================
        # 6. Get user
        # =========================
        user = await User.get(user_id)

        if not user:
            await matched_token.delete()

            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "User not found"}
            )


        # =========================
        # 7. ROTATION (IMPORTANT)
        # =========================
        await matched_token.delete()

        # =========================
        # 8. Generate new tokens
        # =========================
        new_access = create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(minutes=15)
        )

        new_refresh = create_refresh_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(days=7)
        )

        # =========================
        # 9. Store new refresh
        # =========================
        hashed_refresh = hash_password(new_refresh)

        new_doc = RefreshToken(
            user_id=str(user.id),
            token_hash=hashed_refresh,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )

        await new_doc.insert()

        # =========================
        # 10. Response
        # =========================
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "access_token": new_access,
            "refresh_token": new_refresh
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Token refresh failed"}
        )


@router.post("/resend-otp")
async def resend_otp(payload: ResendOTPSchema):

    try:
        email = payload.email.strip()

        user = await User.find_one(User.email == email)

        if not user:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "User not found"}
            )

        # =========================
        # 2. If already verified
        # =========================
        if user.is_verified:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "status": "verified",
                    "action": "LOGIN",
                    "message": "Account already verified. Please login."
                }
            )

        # =========================
        # 3. Delete old OTPs
        # =========================
        await OTP.find(OTP.email == email).delete()

        # =========================
        # 4. Generate new OTP
        # =========================
        otp_code = str(random.randint(100000, 999999))

        otp = OTP(
            email=email,
            code=otp_code,
            expires_at=datetime.utcnow() + timedelta(minutes=10),
        )

        await otp.insert()

        email_service.send_otp_email(email, otp_code)

        # =========================
        # 6. Response
        # =========================
        return {
            "success": True,
            "status": "pending",
            "action": "VERIFY_OTP",
            "message": "OTP resent successfully"
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Failed to resend OTP"}
        )


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordSchema):

    try:
        email = payload.email

        # =========================
        # 1. Check user
        # =========================
        user = await User.find_one(User.email == email)

        if not user:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "User not found"}
            )

        # =========================
        # 2. Must be verified
        # =========================
        if not user.is_verified:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "status": "pending",
                    "action": "VERIFY_OTP",
                    "message": "Account not verified"
                }
            )

        # =========================
        # 3. Delete old OTPs
        # =========================
        await OTP.find(OTP.email == email).delete()

        # =========================
        # 4. Generate OTP
        # =========================
        otp_code = str(random.randint(100000, 999999))

        otp = OTP(
            email=email,
            code=otp_code,
            expires_at=datetime.utcnow() + timedelta(minutes=10),
            is_used=False
        )

        await otp.insert()

        # =========================
        # 5. Send email
        # =========================
        email_service.send_otp_email(email, otp_code)

        return {
            "success": True,
            "action": "RESET_PASSWORD",
            "message": "OTP sent to email"
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Failed to send OTP"}
        )


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordSchema):

    try:
        email = payload.email
        otp_input = payload.otp
        new_password = payload.new_password

        # =========================
        # 1. Find OTP
        # =========================
        otp_record = await OTP.find_one(
            OTP.email == email,
            OTP.code == otp_input
        )

        if not otp_record:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "Invalid OTP"}
            )

        # =========================
        # 2. Check expiry
        # =========================
        if otp_record.expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "OTP expired"}
            )

        # =========================
        # 3. Get user
        # =========================
        user = await User.find_one(User.email == email)

        if not user:
            raise HTTPException(
                status_code=404,
                detail={"success": False, "message": "User not found"}
            )

        # =========================
        # 4. Update password
        # =========================
        user.password = hash_password(new_password)
        user.updated_at = datetime.utcnow()

        await user.save()

        # =========================
        # 5. Cleanup
        # =========================
        await OTP.find(OTP.email == email).delete()
        await RefreshToken.find(RefreshToken.user_id == str(user.id)).delete()

        # =========================
        # 6. Response
        # =========================
        return {
            "success": True,
            "message": "Password reset successfully. Please login again."
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Password reset failed"}
        )

