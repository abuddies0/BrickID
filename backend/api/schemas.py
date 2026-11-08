"""
Request and response schemas.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class EncryptedCredentialIn(BaseModel):
    """
    A client-encrypted credential payload; plaintext is never accepted by this API.
    """

    credential_name: str = Field(min_length=1, max_length=160)
    nonce: str = Field(min_length=1)
    ciphertext: str = Field(min_length=1)
    sort_order: int = 0


class EncryptedCredentialOut(BaseModel):
    """
    Encrypted credential payload returned to its owner.
    """

    id: UUID
    credential_name: str
    nonce: str
    ciphertext: str
    sort_order: int
    updated_at: datetime


class BrickAddedOut(BaseModel):
    """
    Result of adding a backup authenticator.
    """

    brick_id: UUID
    model_version: str


class AuthOut(BaseModel):
    """
    Successful brick authentication result.
    """

    session_token: str
    expires_at: datetime
    user_id: UUID
    credentials: list[EncryptedCredentialOut]
