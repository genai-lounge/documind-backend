"""Backend settings, read once from environment variables at import time.

Every setting has a working default, so a `.env` file is optional in Phase 1.
"""

import os

# Largest accepted upload, in megabytes.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Browser origins allowed to call the API (comma-separated).
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3008").split(",")
    if origin.strip()
]
