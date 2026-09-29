"""REST API (install with ``pip install 'ayurner[server]'``)."""

from .app import create_app, get_app

__all__ = ["create_app", "get_app"]
