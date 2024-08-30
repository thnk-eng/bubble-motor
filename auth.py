# server/auth.py

import os
from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader

def no_auth():
    """
    Placeholder function for no authentication. Use this when no authentication is required.
    """
    pass

def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    """
    Authenticates requests using an API key. If the key is incorrect, an HTTP 401 error is raised.
    """
    expected_api_key = os.environ.get("BUBBLE_MOTOR_API_KEY")
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
