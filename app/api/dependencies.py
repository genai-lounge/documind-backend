"""FastAPI dependencies: how routes receive the use cases they need.

Routes declare what they need, e.g.

    ingest: Annotated[IngestDocument, Depends(get_ingest_document)]

and FastAPI calls these functions to provide it. The objects themselves are
built once at startup by the composition root (app/container.py) and stored
on `app.state`. In tests, `app.dependency_overrides[get_ingest_document]` swaps
in a different instance without touching any route code.
"""

from fastapi import Request

from app.application.compare_texts import CompareTexts
from app.application.ingest_document import IngestDocument


def get_ingest_document(request: Request) -> IngestDocument:
    use_case: IngestDocument = request.app.state.ingest_document
    return use_case


def get_compare_texts(request: Request) -> CompareTexts:
    use_case: CompareTexts = request.app.state.compare_texts
    return use_case
