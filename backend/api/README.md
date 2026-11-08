# FastAPI route layer

This package adds a `/api` prefix:
- `GET /api/health`
- `POST /api/register` (multipart field `files`, repeated)
- `POST /api/authenticate` (multipart field `file`)
- `GET /api/credentials` (Bearer session)
- `PUT /api/credentials` (Bearer session; encrypted payload only)
- `DELETE /api/credentials/{credential_id}` (Bearer session)
- `POST /api/logout` (Bearer session)
- `POST /api/account/bricks` (Bearer session; repeated multipart `files`)

## Required integration notes
- `data/db.py` and `data/models.py` should be the package versions included here, or equivalent models with matching attributes.
- `vision.detector.detect_bricks(image)` is used as-is.
- The identification module must implement `embed_brick(image, bbox) -> 1-D vector` for enrollment and `identify_brick(image, bbox) -> (brick_id, confidence)` for sign-in. Until then those endpoints return HTTP 503 after the detector checks.
- Credentials are ciphertext+nonce. The API never accepts or returns plaintext credential values.
- Add `psycopg[binary]` to `backend` dependencies.

## Transactions
Routes commit their own write transaction and roll back on exceptions. `get_db()` closes the request session.
