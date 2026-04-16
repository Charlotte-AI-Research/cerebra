from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .logging_utils import get_logger

log = get_logger("rag.embed_gateway")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


INTERNAL_BASE_URL = os.getenv("VLLM_INTERNAL_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
GATEWAY_API_KEY = os.getenv("EMBED_GATEWAY_API_KEY")  # optional; if set, required
ALLOW_CORS = _env_bool("EMBED_GATEWAY_CORS", default=True)
MAX_BODY_BYTES = int(os.getenv("EMBED_GATEWAY_MAX_BODY_BYTES", "10485760"))  # 10MB


app = FastAPI(title="Cerebra Embedding Gateway", version="0.1.0")


@app.middleware("http")
async def _guard_and_log(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"error": "request_too_large"})
    resp = await call_next(request)
    return resp


@app.middleware("http")
async def _cors(request: Request, call_next):
    if not ALLOW_CORS:
        return await call_next(request)

    if request.method == "OPTIONS":
        r = JSONResponse(status_code=204, content=None)
    else:
        r = await call_next(request)

    r.headers["Access-Control-Allow-Origin"] = os.getenv("EMBED_GATEWAY_CORS_ORIGIN", "*")
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Authorization,Content-Type"
    return r


def _check_api_key(authorization: Optional[str]) -> None:
    if not GATEWAY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != GATEWAY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


async def _proxy(method: str, path: str, *, authorization: Optional[str], body: Any | None = None):
    _check_api_key(authorization)

    url = f"{INTERNAL_BASE_URL}{path}"
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.request(method, url, json=body)
        except httpx.RequestError:
            log.exception("Failed to reach internal vLLM server", extra={"extra": {"url": url}})
            raise HTTPException(status_code=502, detail="Embedding server unavailable")

    if resp.status_code >= 400:
        # Try to pass through any useful error payload.
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": resp.text}
        log.warning(
            "Internal vLLM error",
            extra={"extra": {"status_code": resp.status_code, "path": path}},
        )
        return JSONResponse(status_code=resp.status_code, content=payload)

    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/healthz")
async def healthz():
    return {"ok": True, "internal_base_url": INTERNAL_BASE_URL}


# OpenAI-compatible endpoints (subset)
@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(default=None)):
    return await _proxy("GET", "/models", authorization=authorization)


@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: Optional[str] = Header(default=None)):
    body = await request.json()
    return await _proxy("POST", "/embeddings", authorization=authorization, body=body)


def main():
    """
    Run with:
      uv run python -m rag.embed_gateway
    Or:
      uv run uvicorn rag.embed_gateway:app --host 0.0.0.0 --port 8002
    """
    import uvicorn

    host = os.getenv("EMBED_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("EMBED_GATEWAY_PORT", "8002"))
    log.info("Starting embedding gateway", extra={"extra": {"host": host, "port": port, "internal": INTERNAL_BASE_URL}})
    uvicorn.run("rag.embed_gateway:app", host=host, port=port, log_level=os.getenv("CEREBRA_LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()

