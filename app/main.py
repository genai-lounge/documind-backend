from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import config
from app.routes import documents

app = FastAPI(title="DocuMind API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(documents.router)


@app.exception_handler(RequestValidationError)
def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return request-validation errors (e.g. no file sent) in the API's standard error shape."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_request",
                "message": "The request is missing a required field or has an invalid value.",
            }
        },
    )


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
