"""API-key auth. Keys are stored as SHA-256 hashes; the plaintext is shown once at creation."""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, status
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import ApiKey
from app.db.session import rw_session


@dataclass(frozen=True)
class Principal:
    key_id: int | None
    name: str


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> str:
    return "rk_" + secrets.token_urlsafe(32)


def create_api_key(name: str) -> str:
    key = generate_key()
    with rw_session() as s:
        s.add(ApiKey(name=name, key_hash=hash_key(key), active=True))
    return key


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Principal:
    if not get_settings().require_api_key:
        return Principal(None, "anonymous")
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key header")
    with rw_session() as s:
        row = s.execute(select(ApiKey).where(ApiKey.key_hash == hash_key(x_api_key), ApiKey.active)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    return Principal(row.id, row.name)
