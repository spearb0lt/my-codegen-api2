# CodeGen API

FastAPI server to convert problem zip -> generated solution / run-only execution.
Set env var GOOGLE_API_KEY to enable LLM calls.
Start: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`


this one is for manual code testeting like past test case as
curl.exe -X POST "https://<your-host>/test" `
  -F "solution_id=b4e7e1de-3f6f-4c53-bd34-1d5cc7a623e3" `
  -F "test_input=2 3"
API Health Check
<img width="895" height="29" alt="image" src="https://github.com/user-attachments/assets/99cd1d85-16c1-4259-b1d5-4dac271a9a3b" />
