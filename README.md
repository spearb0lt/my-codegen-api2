# CodeGen API

FastAPI server to convert problem zip -> generated solution / run-only execution.
Set env var GOOGLE_API_KEY to enable LLM calls.
Start: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`