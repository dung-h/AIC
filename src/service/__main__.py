"""Run with ``python -m src.service``."""
import uvicorn

uvicorn.run("src.service.app:app", host="127.0.0.1", port=8000, reload=False)
