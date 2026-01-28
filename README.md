# This was the initial prototype API I had developed. Please see the main branch for current one


FastAPI server to convert problem zip -> generated solution / run-only execution.
Set env var GOOGLE_API_KEY to enable LLM calls.
Start: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`


this one is for manual code testeting like past test case as
curl.exe -X POST "https://<your-host>/test" `
  -F "solution_id=b4e7e1de-3f6f-4c53-bd34-1d5cc7a623e3" `
  -F "test_input=2 3"
