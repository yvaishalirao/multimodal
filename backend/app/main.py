from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.documents import router as documents_router
from app.llm_provider import validate_llm_config
from app.storage import ensure_bucket


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup on a non-free-tier model rather than on the first call.
    validate_llm_config()
    ensure_bucket()
    yield


app = FastAPI(title="Document RAG + Extraction Pipeline", lifespan=lifespan)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
