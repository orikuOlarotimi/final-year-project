# app/schemas/auth.py

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterSchema(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=2, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str):
        if not v:
            raise ValueError("Email is required")

        v = v.strip().lower()

        if " " in v:
            raise ValueError("Email cannot contain spaces")

        return v

    # ---------------------------
    # PASSWORD VALIDATION
    # ---------------------------
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str):
        if not v:
            raise ValueError("Password is required")

        v = v.strip()

        if len(v) == 0:
            raise ValueError("Password cannot be empty or spaces only")

        return v

    # ---------------------------
    # NAME VALIDATION
    # ---------------------------
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str):
        if not v:
            raise ValueError("Name is required")

        v = v.strip()

        if len(v) == 0:
            raise ValueError("Name cannot be empty or spaces only")

        return v


class LoginSchema(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: EmailStr):
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str):
        v = v.strip()

        if not v:
            raise ValueError("Password is required")

        return v

class VerifyOTPSchema(BaseModel):
    email: EmailStr
    otp: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str):
        if not v:
            raise ValueError("Email is required")

        v = v.strip().lower()

        if " " in v:
            raise ValueError("Email cannot contain spaces")

        return v
    @field_validator("otp")
    @classmethod
    def validate_otp(cls, v: str):
        v = v.strip()

        if not v:
            raise ValueError("OTP is required")

        if not v.isdigit():
            raise ValueError("OTP must contain only digits")

        if len(v) != 6:
            raise ValueError("OTP must be 6 digits")

        return v


class RefreshTokenSchema(BaseModel):
    refresh_token: str

    @field_validator("refresh_token")
    @classmethod
    def validate_token(cls, v: str):
        v = v.strip()

        if not v:
            raise ValueError("Refresh token is required")

        return v

class ResendOTPSchema(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: EmailStr):
        return v.strip().lower()

class ForgotPasswordSchema(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: EmailStr):
        return v.strip().lower()


class ResetPasswordSchema(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: EmailStr):
        return v.strip().lower()

    @field_validator("otp")
    @classmethod
    def validate_otp(cls, v: str):
        v = v.strip()

        if not v or not v.isdigit() or len(v) != 6:
            raise ValueError("Invalid OTP")

        return v

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str):
        v = v.strip()

        if not v or len(v) < 6:
            raise ValueError("Password must be at least 6 characters")

        return v
