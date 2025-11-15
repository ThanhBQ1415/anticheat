import os
from typing import Optional

from fastapi import Depends, Header, HTTPException, status


def get_expected_token() -> Optional[str]:
    return os.getenv("ANTICHEAT_BEARER_TOKEN")


def require_bearer_auth(
    authorization: Optional[str] = Header(None),
    expected_token: Optional[str] = Depends(get_expected_token),
):
    # If no token configured, allow all (dev mode)
    if not expected_token:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1]
    if token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")


def require_session_id(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    return x_session_id


