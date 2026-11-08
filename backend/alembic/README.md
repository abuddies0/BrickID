# Alembic

Run from `backend/`:

```sh
uv add psycopg[binary]
uv run alembic upgrade head
```

For a new revision after modifying ORM models:

```sh
uv run alembic revision --autogenerate -m "describe change"
```

Review generated migrations before applying them. Set `DATABASE_URL` in the environment;
the value in `alembic.ini` is a local development fallback.
