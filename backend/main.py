"""FastAPI application entry point for BrickID."""

from contextlib import asynccontextmanager

from api.routes import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision import detector, identifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vision models when available and release them at shutdown."""
    detector.load_models()
    # The identification model is currently under development; don't prevent API startup
    # if its loader explicitly reports that it is not implemented yet.
    try:
        identifier.load_models()
    except NotImplementedError:
        pass
    yield
    try:
        identifier.cleanup_models()
    except (NotImplementedError, AttributeError):
        pass
    detector.cleanup_models()


app = FastAPI(title="BrickID API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
