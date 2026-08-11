"""Re-export the FastAPI app so auto-detect of `main.py` works on FastAPI Cloud."""

from app.main import app

__all__ = ["app"]
