import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7
BREVO_API_KEY = os.getenv("BREVO_API_KEY")

EMAIL_FROM =  os.getenv("EMAIL_FROM")