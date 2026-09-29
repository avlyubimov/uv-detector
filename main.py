from contextlib import asynccontextmanager
from functools import partial
import hmac
import os
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from detector import AnalysisError, MAX_FILE_BYTES
from library import analyze_photo, load_libraries
from models import AnalysisResponse, ColorLibrary, DEFAULT_LIBRARY_ID, ErrorResponse, LibraryId, Region


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_FILE_BYTES + 64 * 1024:
                raise HTTPException(413, detail={"code": "file_too_large", "message": "Максимальный размер фото — 10 МБ."})
            return message

        await self.app(scope, limited_receive, send)


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_libraries()
    application.state.analysis_limiter = anyio.CapacityLimiter(2)
    yield


app = FastAPI(
    title="UV Detector API", version="0.1.0", lifespan=lifespan,
    description="Школьный проект: сопоставление цвета TEST AREA с учебной таблицей из 101 цвета. По умолчанию используется document_template; калиброванные данные можно заменить в библиотеке.",
)
app.add_middleware(UploadLimitMiddleware)


async def verify_api_key(x_api_key: Annotated[str | None, Header()] = None):
    expected = os.getenv("API_KEY", "")
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(401, detail={"code": "unauthorized", "message": "Неверный X-API-Key."})


@app.exception_handler(AnalysisError)
async def analysis_error_handler(_request: Request, error: AnalysisError):
    return JSONResponse(status_code=error.status_code, content={"detail": {"code": error.code, "message": str(error)}})


@app.get("/")
async def root():
    return {
        "service": "uv-detector", "docs": "/docs", "analyze": "/analyze",
        "default_library": DEFAULT_LIBRARY_ID,
        "measurement_validated": load_libraries()[DEFAULT_LIBRARY_ID].calibrated,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "libraries": len(load_libraries())}


@app.get("/libraries", response_model=list[ColorLibrary], dependencies=[Depends(verify_api_key)])
async def libraries():
    return list(load_libraries().values())


def parse_region(value: str | None, field: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        return Region.model_validate_json(value).as_tuple()
    except ValidationError as error:
        raise AnalysisError("invalid_region", f"{field}: нужен JSON с x, y, width, height в долях изображения (0–1), без выхода за границы.") from error


@app.post(
    "/analyze", response_model=AnalysisResponse, dependencies=[Depends(verify_api_key)],
    responses={status: {"model": ErrorResponse} for status in (400, 401, 413, 415)},
)
async def analyze(
    request: Request,
    file: Annotated[UploadFile, File(description="Фото JPEG, PNG, WebP или BMP, до 10 МБ и 24 Мп")],
    library_id: Annotated[LibraryId, Form()] = DEFAULT_LIBRARY_ID,
    reference_intensity_uw_cm2: Annotated[float | None, Form(gt=0, le=1_000_000, allow_inf_nan=False)] = None,
    roi: Annotated[str | None, Form(description='Необязательная область TEST AREA: {"x":0.1,"y":0.5,"width":0.8,"height":0.2}; координаты после EXIF-поворота')] = None,
    white_roi: Annotated[str | None, Form(description="Необязательная белая область: JSON той же формы, что roi")] = None,
):
    try:
        content = await file.read(MAX_FILE_BYTES + 1)
    finally:
        await file.close()
    operation = partial(
        analyze_photo, content, library_id, reference_intensity_uw_cm2,
        parse_region(roi, "roi"), parse_region(white_roi, "white_roi"),
    )
    return await anyio.to_thread.run_sync(operation, limiter=request.app.state.analysis_limiter)
