"""Database service functions for BrickID.

These functions accept a SQLAlchemy Session. HTTP concerns and response shaping belong
in FastAPI route handlers, not in this module.
"""
import hashlib
import secrets
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Brick, Credential, User
from .models import Session as LoginSession

SESSION_LIFETIME = timedelta(days=7)

def _utcnow() -> datetime:
    """Return a timezone-aware current UTC timestamp."""
    return datetime.now(UTC)

def create_user_with_brick(
    db: Session,
    embedding: Sequence[float],
    model_version: str,
) -> tuple[User, Brick]:
    """Create an account and its first brick in one transaction."""
    user = User()
    brick = Brick(
        user=user,
        embedding=[float(value) for value in embedding],
        model_version=model_version,
    )
    db.add(user)
    db.add(brick)
    db.flush()
    return user, brick

def add_brick_to_user(
    db: Session,
    user_id: uuid.UUID,
    embedding: Sequence[float],
    model_version: str,
) -> Brick | None:
    """Add a backup brick for an existing user, or return None if user is absent."""
    user = db.get(User, user_id)
    if user is None:
        return None
    brick = Brick(
        user=user,
        embedding=[float(value) for value in embedding],
        model_version=model_version,
    )
    db.add(brick)
    db.flush()
    return brick

def get_user_by_brick_id(db: Session, brick_id: uuid.UUID) -> User | None:
    """Resolve the owner of a known brick identifier."""
    stmt = select(User).join(Brick, Brick.user_id == User.id).where(Brick.id == brick_id)
    return db.scalar(stmt)

def get_brick_by_id(db: Session, brick_id: uuid.UUID) -> Brick | None:
    """Fetch a brick enrollment by its primary key."""
    return db.get(Brick, brick_id)

def find_bricks_by_embedding(
    db: Session,
    query_embedding: Sequence[float],
    *,
    model_version: str | None = None,
    limit: int = 10,
) -> list[tuple[Brick, float]]:
    """Return candidate bricks ordered by cosine similarity to a query embedding.

    Similarity is a candidate-ranking score, not a calibrated probability. Final acceptance
    thresholding must be determined from the held-out physical-brick evaluation dataset.
    """
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query.ndim != 1 or query.size == 0 or query_norm == 0 or not np.isfinite(query).all():
        return []

    stmt = select(Brick)
    if model_version is not None:
        stmt = stmt.where(Brick.model_version == model_version)

    matches: list[tuple[Brick, float]] = []
    for brick in db.scalars(stmt):
        vector = np.asarray(brick.embedding, dtype=np.float32)
        if vector.shape != query.shape or not np.isfinite(vector).all():
            continue
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0:
            continue
        score = float(np.dot(query, vector) / (query_norm * vector_norm))
        matches.append((brick, score))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[:max(1, limit)]

def get_credentials_for_user(db: Session, user_id: uuid.UUID) -> list[Credential]:
    """Get a user's encrypted credentials in saved display order."""
    stmt = select(Credential).where(Credential.user_id == user_id).order_by(
        Credential.sort_order.asc(), Credential.credential_name.asc()
    )
    return list(db.scalars(stmt))

def get_credential_for_user(
    db: Session,
    user_id: uuid.UUID,
    credential_name: str,
) -> Credential | None:
    """Get one encrypted credential by its user-scoped name."""
    stmt = select(Credential).where(
        Credential.user_id == user_id,
        Credential.credential_name == credential_name,
    )
    return db.scalar(stmt)

def upsert_credential(
    db: Session,
    user_id: uuid.UUID,
    credential_name: str,
    nonce: bytes,
    ciphertext: bytes,
    sort_order: int = 0,
) -> Credential:
    """Create or update a user's encrypted credential record."""
    existing = get_credential_for_user(db, user_id, credential_name)
    if existing is None:
        existing = Credential(
            user_id=user_id,
            credential_name=credential_name,
            nonce=nonce,
            ciphertext=ciphertext,
            sort_order=sort_order,
        )
        db.add(existing)
    else:
        existing.nonce = nonce
        existing.ciphertext = ciphertext
        existing.sort_order = sort_order
        existing.updated_at = _utcnow()
    db.flush()
    return existing

def create_session(db: Session, user_id: uuid.UUID) -> tuple[str, LoginSession]:
    """Create a random one-week bearer token and store only its SHA-256 hash."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    session = LoginSession(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=_utcnow() + SESSION_LIFETIME,
    )
    db.add(session)
    db.flush()
    return token, session

def get_active_session(db: Session, token: str) -> LoginSession | None:
    """Return a valid, non-revoked session corresponding to a bearer token."""
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    stmt = select(LoginSession).where(
        LoginSession.token_hash == token_hash,
        LoginSession.revoked_at.is_(None),
        LoginSession.expires_at > _utcnow(),
    )
    return db.scalar(stmt)

def revoke_session(db: Session, token: str) -> bool:
    """Revoke a session token; return False if it is unknown or already invalid."""
    session = get_active_session(db, token)
    if session is None:
        return False
    session.revoked_at = _utcnow()
    db.flush()
    return True
