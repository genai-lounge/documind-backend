"""Request middleware: request IDs, access logging, and the last-resort error handler.

Middleware wraps every request. This one:
1. Gives the request an ID (or reuses a safe one sent by the caller in the
   X-Request-ID header), so every log line for it can be found together.
2. Returns that ID in the X-Request-ID response header, so a user can quote
   it when reporting a problem.
3. Logs one line per request with method, path, status, and duration.
4. Catches any unexpected exception, logs the full stack trace, and returns
   a generic 500 error, so users never see internal details.
"""

import logging
import re
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.errors import error_body
from app.core.logging_config import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("request.failed")
                response = JSONResponse(
                    status_code=500,
                    content=error_body(
                        "internal_error", "An unexpected error occurred."
                    ),
                )

            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "request.completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            return response
        finally:
            request_id_var.reset(token)
