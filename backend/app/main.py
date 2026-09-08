from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.documents import router as documents_router
from app.storage import ensure_bucket


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_bucket()
    yield


app = FastAPI(title="Document RAG + Extraction Pipeline", lifespan=lifespan)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
