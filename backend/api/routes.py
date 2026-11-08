"""BrickID API routes for enrollment, brick authentication, credentials, and logout."""

import base64
import binascii
from typing import Annotated
from uuid import UUID

import numpy as np
from data.models import Session as LoginSession
from data.services import (
    add_brick_to_user,
    create_session,
    create_user_with_brick,
    get_credentials_for_user,
    get_user_by_brick_id,
    revoke_session,
    upsert_credential,
)
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from vision import detector, identifier
from vision.identifier import bytes_to_image

from api.dependencies import get_current_session, get_db
from api.schemas import (
    AuthOut,
    BrickAddedOut,
    EncryptedCredentialIn,
    EncryptedCredentialOut,
)

router = APIRouter(prefix="/api", tags=["BrickID"])
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MODEL_VERSION = "siamese-v1"


async def _read_image(upload: UploadFile):
    """
    Validate and decode an uploaded image into the project's OpenCV image format.

    :param upload: The uploaded file to validate
    :return: The image as an OpenCV MatLike
    """
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422, detail="Only JPEG, PNG, and WEBP images are supported."
        )
    contents = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB limit.")
    image = bytes_to_image(contents)
    if image is None:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file could not be decoded as an image.",
        )
    return image


def _largest_box(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """
    Return the largest detection box, or None when the image contains no brick.

    :param boxes: A list of bounding boxes returned by the detector.
    :return: The largest bounding box
    """
    if not boxes:
        return None
    return max(boxes, key=lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _extract_embedding(image, box) -> list[float]:
    """
    Call the future identifier embedding interface and validate its numeric output.

    :param image: The OpenCV image to extract embeddings from.
    :param box: The bounding box of the brick that was detected.

    :return: The embeddings associated with the given brick.
    """
    embed = getattr(identifier, "embed_brick", None)
    if embed is None:
        raise HTTPException(
            status_code=503, detail="Brick enrollment model is not available yet."
        )
    try:
        result = embed(image, box)
        vector = np.asarray(result, dtype=np.float32)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=503, detail="Brick enrollment model is not available yet."
        ) from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise HTTPException(
            status_code=422, detail="The model did not produce a valid brick embedding."
        )
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise HTTPException(
            status_code=422, detail="The model produced an empty embedding."
        )
    return (vector / norm).astype(float).tolist()


def _decode_base64(value: str, field_name: str) -> bytes:
    """
    Decode strict base64 supplied by the browser.

    :param value: The base 64 string
    :param field_name: The fieldname in the base 64 string

    :return: The bytes in the base64 string
    """
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{field_name} must be valid base64."
        ) from exc


def _credential_out(record) -> EncryptedCredentialOut:
    """
    Convert a database credential record to a JSON-safe response.
    """
    return EncryptedCredentialOut(
        id=record.id,
        credential_name=record.credential_name,
        nonce=base64.b64encode(record.nonce).decode("ascii"),
        ciphertext=base64.b64encode(record.ciphertext).decode("ascii"),
        sort_order=record.sort_order,
        updated_at=record.updated_at,
    )


@router.get("/health")
def health() -> dict[str, str]:
    """
    Return a simple API health response.
    """
    return {"status": "ok"}


@router.post("/register", status_code=201)
async def register_brick(
    files: Annotated[
        list[UploadFile],
        File(description="Multiple enrollment images of the same brick."),
    ],
    # Temporarily suppressing this warning since I'm using a custom generator for get_db
    db: Session = Depends(get_db),  # noqa: B008
) -> dict:
    """
    Register a new user using several images of one brick.

    Enrollment is intentionally unavailable until the Siamese model exposes embed_brick().
    """
    if len(files) < 3:
        raise HTTPException(
            status_code=422, detail="Upload at least 3 images of the same brick."
        )
    if len(files) > 30:
        raise HTTPException(
            status_code=422, detail="Upload no more than 30 enrollment images."
        )

    vectors: list[np.ndarray] = []
    detected_count = 0
    for upload in files:
        image = await _read_image(upload)
        boxes = detector.detect_bricks(image)
        box = _largest_box(boxes)
        if box is None:
            continue
        detected_count += 1
        vectors.append(np.asarray(_extract_embedding(image, box), dtype=np.float32))

    if detected_count == 0:
        raise HTTPException(
            status_code=422,
            detail="Upload processed, but no brick was found in any image.",
        )
    if len(vectors) < 3:
        raise HTTPException(
            status_code=422,
            detail="A brick was detected in fewer than 3 images; please capture more clear views.",
        )

    # Normalize the mean embedding so enrollment represents the submitted views.
    mean_vector = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean_vector))
    if norm == 0 or not np.isfinite(mean_vector).all():
        raise HTTPException(
            status_code=422,
            detail="The enrollment images did not produce a consistent brick embedding.",
        )
    mean_vector = mean_vector / norm

    try:
        user, brick = create_user_with_brick(
            db, mean_vector.astype(float).tolist(), MODEL_VERSION
        )
        db.commit()
    # Warning suppressed because I want it to catch ANY error
    except Exception:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Could not create the account. Please try again."
        )

    return {
        "user_id": str(user.id),
        "brick_id": str(brick.id),
        "registered_images": len(vectors),
        "model_version": MODEL_VERSION,
    }


@router.post("/authenticate", response_model=AuthOut)
async def authenticate_brick(
    file: Annotated[UploadFile, File()],
    db: Session = Depends(get_db),  # noqa: B008
) -> AuthOut:
    """
    Authenticate a brick and return its user's encrypted vault payloads.
    """
    image = await _read_image(file)
    boxes = detector.detect_bricks(image)
    box = _largest_box(boxes)
    if box is None:
        raise HTTPException(
            status_code=404, detail="Upload successful, but no brick was found."
        )

    identify = getattr(identifier, "identify_brick", None)
    if identify is None:
        raise HTTPException(
            status_code=503, detail="Brick identification model is not available yet."
        )
    try:
        brick_id, _confidence = identify(image, box)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=503, detail="Brick identification model is not available yet."
        ) from exc
    if brick_id is None:
        raise HTTPException(
            status_code=401,
            detail="A brick was detected, but it could not be identified.",
        )
    try:
        resolved_id = UUID(str(brick_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="The identification model returned an invalid brick ID.",
        ) from exc

    user = get_user_by_brick_id(db, resolved_id)
    if user is None:
        raise HTTPException(status_code=401, detail="The brick is not registered.")
    try:
        token, session = create_session(db, user.id)
        records = get_credentials_for_user(db, user.id)
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Could not create an authentication session."
        )

    return AuthOut(
        session_token=token,
        expires_at=session.expires_at,
        user_id=user.id,
        credentials=[_credential_out(record) for record in records],
    )


@router.get("/credentials", response_model=list[EncryptedCredentialOut])
def list_credentials(
    current: LoginSession = Depends(get_current_session),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> list[EncryptedCredentialOut]:
    """
    Return the current user's encrypted credential records.
    """
    return [
        _credential_out(item) for item in get_credentials_for_user(db, current.user_id)
    ]


@router.put("/credentials", response_model=EncryptedCredentialOut)
def save_credential(
    payload: EncryptedCredentialIn,
    current: LoginSession = Depends(get_current_session),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> EncryptedCredentialOut:
    """
    Save an opaque encrypted credential payload for the authenticated user.
    """
    nonce = _decode_base64(payload.nonce, "nonce")
    ciphertext = _decode_base64(payload.ciphertext, "ciphertext")
    if len(nonce) != 12 or len(ciphertext) < 16:
        raise HTTPException(
            status_code=422, detail="Invalid AES-GCM nonce or ciphertext."
        )
    try:
        record = upsert_credential(
            db,
            current.user_id,
            payload.credential_name.strip(),
            nonce,
            ciphertext,
            payload.sort_order,
        )
        db.commit()
        db.refresh(record)
    except Exception:  # noqa: BLE001
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not save the credential.")
    return _credential_out(record)


@router.delete("/credentials/{credential_id}", status_code=204)
def delete_credential(
    credential_id: UUID,
    current: LoginSession = Depends(get_current_session),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
):
    """
    Delete one credential owned by the current user.
    """
    from data.models import Credential
    from sqlalchemy import select

    record = db.scalar(
        select(Credential).where(
            Credential.id == credential_id, Credential.user_id == current.user_id
        )
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Credential not found.")
    db.delete(record)
    db.commit()


@router.post("/logout", status_code=204)
def logout(
    credentials=Depends(  # noqa: B008
        __import__("api.dependencies", fromlist=["bearer_scheme"]).bearer_scheme  # noqa: B008
    ),
    db: Session = Depends(get_db),  # noqa: B008
):
    """
    Revoke the current bearer session.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required.")
    if not revoke_session(db, credentials.credentials):
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    db.commit()


@router.post("/account/bricks", response_model=BrickAddedOut, status_code=201)
async def add_backup_brick(
    files: Annotated[
        list[UploadFile], File(description="Images for the backup brick.")
    ],
    current: LoginSession = Depends(get_current_session),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> BrickAddedOut:
    """
    Enroll a backup brick for the authenticated account.
    """
    if len(files) < 3 or len(files) > 30:
        raise HTTPException(status_code=422, detail="Upload between 3 and 30 images.")
    vectors: list[np.ndarray] = []
    for upload in files:
        image = await _read_image(upload)
        box = _largest_box(detector.detect_bricks(image))
        if box is not None:
            vectors.append(np.asarray(_extract_embedding(image, box), dtype=np.float32))
    if len(vectors) < 3:
        raise HTTPException(
            status_code=422, detail="A brick must be detected in at least 3 images."
        )
    mean_vector = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean_vector))
    if norm == 0 or not np.isfinite(mean_vector).all():
        raise HTTPException(
            status_code=422, detail="The images did not produce a consistent embedding."
        )
    try:
        brick = add_brick_to_user(
            db,
            current.user_id,
            (mean_vector / norm).astype(float).tolist(),
            MODEL_VERSION,
        )
        if brick is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:  # noqa: BLE001
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not register backup brick.")
    return BrickAddedOut(brick_id=brick.id, model_version=brick.model_version)
