"""Entry point for hosts that import a ready-made ASGI app, such as Vercel.

Locally, `make dev` uses the `create_app` factory instead.
"""

from app.main import create_app

app = create_app()
