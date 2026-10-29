"""Chooses where original files are stored: FILE_STORAGE=local or supabase.

Part of the composition root. Configuration and connection problems become
ConfigurationError, so the app refuses to start with a clear message.
"""

import logging

from app.application.ports import FileStorage
from app.core.config import ConfigurationError, Settings
from app.infrastructure.storage.local_file_storage import LocalFileStorage
from app.infrastructure.storage.supabase_file_storage import SupabaseFileStorage

logger = logging.getLogger(__name__)


def build_file_storage(settings: Settings) -> FileStorage:
    if settings.file_storage == "local":
        return LocalFileStorage(settings.uploads_dir)

    url = (settings.supabase_url or "").strip()
    key = (
        settings.supabase_secret_key.get_secret_value().strip()
        if settings.supabase_secret_key
        else ""
    )
    if not url.startswith("https://"):
        raise ConfigurationError(
            "SUPABASE_URL is not set (e.g. https://abcd.supabase.co). Find it in your "
            "Supabase project under Project Settings → API, or set FILE_STORAGE=local."
        )
    if not key:
        raise ConfigurationError(
            "SUPABASE_SECRET_KEY is not set. Use the project's secret key "
            "(sb_secret_...) from Project Settings → API Keys, in backend/.env only."
        )
    try:
        storage = SupabaseFileStorage(url, key, settings.supabase_bucket)
        created = storage.ensure_bucket()
    except Exception as exc:
        raise ConfigurationError(
            "Could not use Supabase Storage. Check SUPABASE_URL, SUPABASE_SECRET_KEY, "
            "and SUPABASE_BUCKET, and that the project isn't paused. "
            f"Details: {type(exc).__name__}"
        ) from exc
    logger.info(
        "file_storage.connected",
        extra={
            "provider": "supabase",
            "bucket": storage.bucket,
            "bucket_created": created,
        },
    )
    return storage
