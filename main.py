import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from utils.logger import setup_logging
from stores import qdrant_store, pg_store
from helpers.embeddings import close_http_client
from routes import search_router, opportunities_router

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage connections on startup and teardown."""
    logger.info("Starting up — connecting to stores...")
    await qdrant_store.connect()
    await pg_store.connect()
    logger.info("All stores connected ✓")

    yield  # App is running

    logger.info("Shutting down — closing connections...")
    await qdrant_store.close()
    await pg_store.close()
    await close_http_client()
    logger.info("Shutdown complete")


# ──────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────

app = FastAPI(
    title="ScholarX Search API",
    description="Semantic search API for scholarships and opportunities",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: allow all origins (file://, localhost, etc.)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(search_router)
app.include_router(opportunities_router)


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


# ──────────────────────────────────────────────
#  Run with: uvicorn main:app --reload
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
    )
