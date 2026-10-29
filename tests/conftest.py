"""Shared pytest fixtures (pytest finds this file automatically)."""

import atexit
import os
import shutil
import tempfile

# Before any test module is imported: point DATA_DIR at a throwaway folder so
# no test can ever touch the real backend/data (your database and vectors).
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="documind-tests-")
os.environ["DATA_DIR"] = _TEST_DATA_DIR
atexit.register(shutil.rmtree, _TEST_DATA_DIR, ignore_errors=True)

import pytest

from app.core.config import Settings
from tests.factories import make_docx, make_pdf


@pytest.fixture
def settings() -> Settings:
    """Default settings, ignoring any local backend/.env so tests are repeatable."""
    return Settings(_env_file=None)


@pytest.fixture
def two_page_pdf() -> bytes:
    return make_pdf(
        ["Alpha page one", "Beta page two"], title="Sample", author="Tester"
    )


@pytest.fixture
def sample_docx() -> bytes:
    return make_docx(
        ["Introduction paragraph", "Second paragraph"],
        table=[["Name", "Score"], ["Ada", "99"]],
        title="Doc Title",
    )
