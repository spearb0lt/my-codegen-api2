# CodeGen API — AI-Powered Competitive Programming Solver

## Built for Meta Hacker Cup 2025 (AI Track)

**Meta Hacker Cup** is Meta's premier global competitive programming contest, attracting tens of thousands of participants worldwide. In 2025, Meta introduced the **AI Track** — a parallel competition where participants could leverage AI systems to solve the same algorithmic problems that top human programmers struggle with. The AI Track permitted any form of AI assistance, from prompt engineering to fully autonomous solvers.

This API was purpose-built as an **end-to-end autonomous solver** for the Meta Hacker Cup AI Track. It accepts a problem package (statement + sample I/O + optional images), generates an optimal Python solution using Google's Gemini LLM, validates it against sample test cases, and iteratively regenerates on failure — all in a single API call. The system was designed to handle the unique characteristics of Hacker Cup problems:

- **Image-heavy problem statements** — Hacker Cup frequently embeds diagrams, graphs, and visual examples (often served from Facebook CDN URLs) directly in problem statements. This API has a complete multimodal pipeline to fetch, decode, and pass these images to the model.
- **Strict I/O format** — Solutions must read from stdin and write exact output (e.g., `Case #1: 42`). The API enforces this by running solutions against sample data before returning them.
- **Multi-attempt regeneration** — If a solution produces wrong output, the API feeds the error back to the model and tries again (up to 4 attempts by default).
- **Real test case debugging** — The `/test` and `/test2` endpoints allow re-running solutions on actual competition test inputs and triggering LLM-based fixes when they fail.

### Competition Results

Using this system (previously with `gemini-2.5-pro`, now you can try with `gemini-3.5-flash`), problems of significant difficulty were solved, including:

| Problem | Round | Points | Estimated Codeforces Rating |
|---------|-------|--------|---------------------------|
| [Designing Paths (C)](https://www.facebook.com/codingcompetitions/hacker-cup/2025/round-2/problems/C) | Round 2 | 23 pts | ~2200–2400 (graph BFS with constrained edge traversal on tram routes) |
| [Treehouse Telegram (D)](https://www.facebook.com/codingcompetitions/hacker-cup/2025/round-3/problems/D) | Round 3 | 24 pts | ~2300–2500 (tree distances + GCD-based pair enumeration using number theory) |

These are problems in the upper-medium to hard range of competitive programming, involving graph algorithms, number theory, and careful complexity analysis.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (you)                                 │
│   Upload .zip (statement + sample_in + sample_out + images)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /generate
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (Render)                          │
│                                                                     │
│  1. Unpack ZIP → extract statement, sample I/O, images              │
│  2. Process images:                                                 │
│     a. Local .png/.jpg → PIL.Image                                  │
│     b. URI .txt files → fetch URL → PIL.Image                       │
│     c. If PIL fails → URIs included as text in prompt               │
│  3. Build prompt with problem + sample I/O + image mapping          │
│  4. Call Gemini 3.5 Flash (thinking_level=high)                     │
│  5. Extract Python code from markdown response                      │
│  6. Run code against sample input                                   │
│  7. Compare output to expected                                      │
│     ├─ MATCH → save & return solution                               │
│     └─ MISMATCH → feed error back to LLM → retry (up to 4x)       │
│  8. Return solution + metadata                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Endpoints

### `GET /health`

Health check. Returns model configuration status, whether PIL/requests are installed.

### `POST /generate`

**Primary endpoint.** Accepts a ZIP file containing the problem and returns a validated solution.

**Parameters:**
- `file` (required): ZIP containing `statement.txt`, `sample_in.txt`, `sample_out.txt`, and optionally image files or image URI text files.
- `check` (optional, default `true`): If `true`, solution must match sample output exactly. Set to `false` to skip validation.

**Workflow:**
1. Unpack ZIP, locate problem files by naming heuristics
2. Process any images (see Image Pipeline below)
3. Construct prompt with problem statement + sample I/O
4. Call Gemini API with `thinking_level="high"` for maximum reasoning
5. Extract Python code from the LLM's markdown response
6. Execute the code with sample input, compare stdout to expected output
7. On success → persist artifacts, return solution
8. On failure → include error details in next prompt, regenerate (up to `MAX_GENERATION_ATTEMPTS`)

**Response (success):**
```json
{
  "status": "generated",
  "attempt": 2,
  "solution_id": "uuid-...",
  "solution": "import sys\n...",
  "sample_stdout": "Case #1: 42\n",
  "raw_llm_text": "...",
  "multimodal_errors": []
}
```

**Response (exhausted):** HTTP 400 with `status: "failed"`, including `last_error` and `last_solution`.

---

### `POST /test`

Re-run a solution on a new test input. If it fails, the LLM regenerates a fix.

**Parameters:**
- `solution_id` (or `solution`/`solution_file`/`problem_zip`): identify or provide the solution
- `test_input` or `test_file`: the new test case
- `test_expected` (optional): expected output for validation
- `check` (optional, default `true`): enable/disable output matching

**Workflow:**
1. Run the existing solution on the provided test input
2. If it passes (no stderr, output matches expected if provided) → return immediately
3. If it fails → construct a regeneration prompt including: problem statement, sample I/O, previous code, failing test input, and the failure details
4. Call Gemini API, validate regenerated code against both sample and test inputs
5. Repeat up to `MAX_GENERATION_ATTEMPTS` times

---

### `POST /test2`

JSON-based endpoint for local-run results. Your local machine runs the code, and if it fails, you POST the failure here for LLM-based regeneration.

**JSON Body:**
```json
{
  "solution_id": "uuid-...",
  "stdout": "...",
  "stderr": "RuntimeError: ...",
  "timed_out": false,
  "returncode": 1,
  "test_input": "5\n1 2 3 4 5",
  "test_expected": "Case #1: 15"
}
```

**Behavior:** If the local run succeeded (no stderr, output matches), returns `status: "ok"`. Otherwise triggers the same regeneration pipeline as `/test`.

---

### `GET /download/{solution_id}/{filename}`

Download any artifact from a solution directory (e.g., `coding_solution.py`, `statement.txt`, `test_output.txt`).

### `GET /solutions/{solution_id}/files`

List all files stored for a given solution.

### `GET /solutions`

List all stored solution IDs and their files.

### `DELETE /cleanup/{solution_id}`

Remove all artifacts for a solution. Safely restricted to the solutions directory.

---

## Image Pipeline (Multimodal)

Meta Hacker Cup problems often include **diagrams and figures** embedded via `{{PHOTO_ID:123|WIDTH:600}}` placeholders that reference Facebook CDN image URLs. This API handles them through a multi-layered fallback system:

### Image Discovery
1. **Direct image files** in the ZIP (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`)
2. **URI text files** (files containing `image` or `photo` in the name with `.txt`/`.uri`/`.url` extensions) — the file content is treated as an image URL

### Image Processing Pipeline
```
Image file found in ZIP
    │
    ├─ Direct image (.png/.jpg) → Open with PIL → Success ✓
    │                                └─ PIL fails → record error
    │
    └─ URI text file (.txt/.uri) → Read URL from file
                                       │
                                       ├─ fetch_image_as_pil(url)
                                       │   ├─ HTTP GET with browser-like headers
                                       │   ├─ Up to 3 retries with backoff
                                       │   ├─ PIL.Image.open(bytes) → Success ✓
                                       │   └─ Fails → record URL in image_urls list
                                       │
                                       └─ URL preserved for text-based fallback
```

### Prompt Construction with Images

**When PIL succeeds:** Image data is available for multimodal prompt parts (currently the code uses text-only mode with URL mapping for reliability).

**When PIL fails (or always, in current code path):** The image URLs are embedded directly into the prompt as text:
```
-- Image mapping (attachment order) --
[IMG1] -> https://scontent.fbcdn.net/v/...
[IMG2] -> https://scontent.fbcdn.net/v/...
```

**Answer to FAQ:** Yes — if PIL fails to fetch/decode an image, the URI is still passed as text in the prompt. The model can use URL context or its training knowledge to reason about the problem even without the actual pixel data. The mapping is always included in the prompt regardless of whether PIL succeeded, ensuring the model knows about all images.

### Placeholder Matching

The system matches `{{PHOTO_ID:X}}` placeholders in the statement to image files by:
1. Checking if the photo ID appears in the filename
2. Checking if the photo ID appears in the content of URI text files
3. Falling back to positional order matching for unmatched placeholders

---

## Regeneration & Failure Pipeline

The system implements an iterative self-correction loop:

```
Attempt 1: Generate from scratch
    │
    ├─ Code runs successfully, output matches → DONE ✓
    │
    └─ Failure detected:
        ├─ Timeout → "Timeout after 300s"
        ├─ Runtime error → stderr captured
        └─ Wrong answer → unified diff of expected vs actual
            │
            ▼
Attempt 2: Regenerate with error context
    Prompt includes:
    - Original problem statement
    - Previous code (full)
    - Specific failure reason (diff/error/timeout)
    - Sample I/O for reference
    │
    ├─ Success → DONE ✓
    └─ Failure → Attempt 3...
            │
            ▼
Attempt 3–4: Same pattern
    │
    └─ All attempts exhausted → Return HTTP 400 with last_solution
```

### What the model receives on retry:
```
Previous submission produced incorrect output.

Problem statement: [full text]
Previous code: [complete Python code]
Reason for failure: [unified diff / stderr / timeout message]
Sample Input: [full sample]
Expected Sample Output: [full expected]

Please provide a corrected most optimized and complete Python solution.
```

---

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `MODEL_NAME` | `gemini-3.5-flash` | Model to use for generation |
| `MAX_GENERATION_ATTEMPTS` | `4` | Max retries per request |
| `EXECUTION_TIMEOUT` | `300` | Seconds before killing a solution run |
| `SOLUTIONS_DIR` | `solutions` | Directory to persist artifacts |
| `ENABLE_IMAGE_DOWNLOAD` | `1` | Enable fetching images from URLs |
| `MAX_IMAGES_IN_PROMPT` | `6` | Cap on images included in prompt |
| `PYTHON_PATH` | `python` | Python executable for running solutions |

---

## Deployment

### On Render (recommended)

1. Push this repo to GitHub
2. Create a new Web Service on Render pointing to the repo
3. Set environment variables: `GOOGLE_API_KEY`, optionally `MODEL_NAME`
4. Start command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

### Local

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-here"
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

---

## Usage Examples

### Generate a solution
```python
import requests, json
from pathlib import Path

url = "https://my-codegen-api2.onrender.com/generate"
files = {"file": ("problem.zip", open("MyQ.zip", "rb"), "application/zip")}
r = requests.post(url, files=files, timeout=600)
j = r.json()
print(j["status"], "attempt:", j.get("attempt"))
Path("coding_solution.py").write_text(j["solution"])
```

### Test with a new input
```python
r = requests.post(
    "https://my-codegen-api2.onrender.com/test",
    data={"solution_id": j["solution_id"]},
    files={"test_file": open("test_input.txt", "rb")}
)
print(r.json()["status"])
```

### PowerShell
```powershell
$Url = "https://my-codegen-api2.onrender.com"
curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
curl.exe -s -X POST "$Url/test" -F "solution_id=<ID>" -F "test_file=@test_input.txt" -o test_response.json
```

---

## ZIP File Format

The input ZIP should contain (naming is flexible — matched by heuristics):

```
MyQ.zip
├── statement.txt          # Problem statement (required)
├── sample_in.txt          # Sample input (required)
├── sample_out.txt         # Sample expected output (required)
├── image_1_uri.txt        # (optional) Contains a URL to a problem diagram
├── photo_2_uri.txt        # (optional) Another image URL
└── diagram.png            # (optional) Direct image file
```

File matching heuristics:
- **Statement:** filename contains `statement` or `problem`
- **Sample input:** filename contains `sample_in`, `sample.in`, `input`, or `sample-input`
- **Sample output:** filename contains `sample_out`, `sample.out`, `output`, or `sample-output`
- **Images:** files with image extensions OR files with `image`/`photo` in the name

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI |
| LLM | Google Gemini 3.5 Flash (via `google-genai` SDK ≥2.5.0) |
| Thinking | `thinking_level="high"` for maximum reasoning depth |
| Image Processing | Pillow (PIL) |
| HTTP Client | requests (for image fetching) |
| Code Execution | subprocess with timeout |
| Deployment | Render / any ASGI server (uvicorn) |

---

## Dependencies

```
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
python-multipart>=0.0.6
google-genai>=2.5.0
packaging
requests
Pillow
```

---

## Setting Up on Render (Step-by-Step)

> FastAPI server to convert problem zip → generated solution / run-only execution.  
> Set env var `GOOGLE_API_KEY` to enable LLM calls.  
> Start: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

### 1. Push your code to GitHub

```bash
git init
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/my-codegen-api2.git
git push -u origin main
```

### 2. Create a Render account

Go to [https://render.com](https://render.com) and sign up (GitHub OAuth recommended for easy repo linking).

### 3. Create a new Web Service

1. From the Render Dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub account if not already connected
3. Select the **`my-codegen-api2`** repository
4. Configure the service:

| Setting | Value |
|---------|-------|
| **Name** | `my-codegen-api2` (or any name you prefer) |
| **Region** | Choose closest to you (e.g., Oregon, Frankfurt) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn api_server:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free (or Starter for faster cold starts) |

### 4. Set environment variables

In the Render service settings, go to **"Environment"** tab and add:

| Key | Value | Notes |
|-----|-------|-------|
| `GOOGLE_API_KEY` | `your-gemini-api-key` | **Required.** Get from [Google AI Studio](https://aistudio.google.com/apikey) |
| `MODEL_NAME` | `gemini-3.5-flash` | Optional. Defaults to `gemini-3.5-flash` |
| `PYTHON_VERSION` | `3.11.6` | Optional. Ensures consistent Python version |

### 5. Deploy

Click **"Create Web Service"**. Render will:
1. Clone your repo
2. Run `pip install -r requirements.txt`
3. Start the server with `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

The service URL will be something like: `https://my-codegen-api2.onrender.com`

### 6. Verify deployment

```bash
curl https://my-codegen-api2.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model": "gemini-3.5-flash",
  "pil_available": true,
  "requests_available": true
}
```

### 7. Auto-deploy on push

By default, Render auto-deploys whenever you push to `main`. To update your service:

```bash
git add -A
git commit -m "Update"
git push
```

Render will automatically rebuild and redeploy within ~1–2 minutes.

### Notes

- **Cold starts:** On the free tier, the service spins down after 15 minutes of inactivity. First request after idle takes ~30–50 seconds to boot.
- **Timeouts:** Render free tier has a 30-second request timeout. For `/generate` (which can take minutes with retries), upgrade to a paid plan or set `MAX_GENERATION_ATTEMPTS=2` to reduce total time.
- **Logs:** View live logs from the Render dashboard → your service → "Logs" tab. Useful for debugging 400/500 errors.
- **Disk:** The `solutions/` directory is ephemeral on Render (lost on redeploy). For persistence, use Render Disks (paid) or treat solutions as temporary.

---

## License

Private repository. Built for Meta Hacker Cup 2025 AI Track participation.
