"""
Robust Code-Gen Two-Step FastAPI server (final merge).
Features:
 - /health
 - /generate : upload problem ZIP (statement, sample_in, sample_out) -> LLM generates code until sample passes.
 - /test     : multiple modes including regeneration with LLM when test input fails.
 - /test2    : JSON endpoint for client-local-run workflow (post stdout/stderr/timed_out etc)
 - /cleanup  : delete saved solution artifacts
 - /download/{solution_id}/{filename} : download allowed artifacts
 - GET /solutions/{solution_id}/files : list saved files
 - GET /solutions : list existing solution_ids and file lists
Notes:
 - This server executes arbitrary Python code via subprocess. This is UNSAFE for untrusted inputs.
 - On ephemeral hosts (Render free tier) disk is not durable — download artifacts immediately client-side.
"""


import os
import shutil
import zipfile
import tempfile
import subprocess
import difflib
import uuid
import html
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
import json
import time

from fastapi.responses import JSONResponse, FileResponse

import re
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.IGNORECASE)

app = FastAPI(title="Code-Gen Two-Step Final API")

# Configuration via environment
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "60"))  # seconds

ALLOWED_DOWNLOAD_FILES = {
    'coding_solution.py', 'test_output.txt', 'sample_stdout.txt',
    'statement.txt', 'sample_in.txt', 'sample_out.txt', 'llm_response.txt', 'metadata.json', 'gen_response.json'
}

def extract_python_from_markdown(text: str) -> Optional[str]:
    """
    Extract the code from a fenced markdown block if present. Otherwise return the entire text.
    Also perform a safe unicode-escape decoding to turn literal "\\n" sequences into real newlines when appropriate.
    """
    if not text:
        return None
    # If there are common escape sequences like "\n" present in the raw text, try to decode them.
    # Use heuristics to avoid double-decoding proper text.
    try:
        if "\\n" in text or "\\t" in text or '\\"' in text:
            try:
                decoded = bytes(text, "utf-8").decode("unicode_escape")
                # prefer decoded if it yields at least as many real newlines
                if decoded.count("\n") >= text.count("\n"):
                    text = decoded
            except Exception:
                pass
    except Exception:
        pass


 
 # try:
    #     if ("\n" in text or "\t" in text or '\"' in text) and text.count("\n") > text.count("\n"):
    #         decoded = bytes(text, "utf-8").decode("unicode_escape")
    #         # prefer decoded if it yields more real newlines
    #         if decoded.count("\n") >= text.count("\n"):
    #             text = decoded
    # except Exception:
    #     # fallback: leave text as-is
    #     pass

    # Try to extract fenced code block
    m = CODE_FENCE_RE.search(text)
    if m:
        code = m.group(1)
    else:
        code = text

    # Some LLMs may return HTML-escaped text; unescape common entities
    code = html.unescape(code)
    # Normalize line endings and strip outer whitespace/newlines
    code = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return code



ASCII_ART = """

██████╗░██╗░░░░░███████╗░█████╗░░██████╗███████╗  ██╗░░██╗██╗██████╗░███████╗  ███╗░░░███╗███████╗
██╔══██╗██║░░░░░██╔════╝██╔══██╗██╔════╝██╔════╝  ██║░░██║██║██╔══██╗██╔════╝  ████╗░████║██╔════╝
██████╔╝██║░░░░░█████╗░░███████║╚█████╗░█████╗░░  ███████║██║██████╔╝█████╗░░  ██╔████╔██║█████╗░░
██╔═══╝░██║░░░░░██╔══╝░░██╔══██║░╚═══██╗██╔══╝░░  ██╔══██║██║██╔══██╗██╔══╝░░  ██║╚██╔╝██║██╔══╝░░
██║░░░░░███████╗███████╗██║░░██║██████╔╝███████╗  ██║░░██║██║██║░░██║███████╗  ██║░╚═╝░██║███████╗
╚═╝░░░░░╚══════╝╚══════╝╚═╝░░╚═╝╚═════╝░╚══════╝  ╚═╝░░╚═╝╚═╝╚═╝░░╚═╝╚══════╝  ╚═╝░░░░░╚═╝╚══════╝
"""





def unpack_zip_to_dir(zip_bytes: bytes, dest_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpf:
        tmpf.write(zip_bytes)
        tmpf.flush()
        tmpf_path = Path(tmpf.name)
    with zipfile.ZipFile(tmpf_path, 'r') as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_name = Path(member.filename).name
            target_path = dest_dir / target_name
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    try:
        tmpf_path.unlink(missing_ok=True)
    except Exception:
        pass

def find_problem_files(workdir: Path) -> Dict[str, Path]:
    file_keys = {
        'statement': ['statement', 'problem'],
        'sample_in': ['sample_in', 'sample.in', 'input', 'sample-input'],
        'sample_out': ['sample_out', 'sample.out', 'output', 'sample-output']
    }
    found = {}
    for p in workdir.iterdir():
        name = p.name.lower()
        for key, patterns in file_keys.items():
            if any(pat in name for pat in patterns):
                found[key] = p
                break
    return found

def run_python_code_str(code_str: str, input_str: str, timeout: int = EXECUTION_TIMEOUT) -> Dict[str, Any]:
    """
    Run code_str with `python -c` in a subprocess and return dict with stdout, stderr, timed_out.
    Note: This is not sandboxed. Use only in controlled environments.
    """
    try:
        p = subprocess.Popen(
            [os.getenv("PYTHON_PATH", "python"), "-c", code_str],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = p.communicate(input=input_str, timeout=timeout)
        return {"stdout": out, "stderr": err, "timed_out": False, "returncode": p.returncode}
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "timed_out": True, "returncode": None}
    except Exception as e:
        return {"stdout": "", "stderr": f"Runtime error: {e}", "timed_out": False, "returncode": None}

def save_solution_on_server(solution_text: str, solution_dir: Path, metadata: Dict[str, Any]) -> Path:
    """
    Save solution text and metadata into the solution_dir.
    Returns path of saved code.
    """
    solution_dir.mkdir(parents=True, exist_ok=True)
    coding_path = solution_dir / 'coding_solution.py'
    solution_text_norm = solution_text.replace("\r\n", "\n").replace("\r", "\n")
    coding_path.write_text(solution_text_norm, encoding='utf-8')
    # store metadata and optionally raw llm response
    (solution_dir / 'metadata.json').write_text(str(metadata), encoding='utf-8')
    if 'raw_llm' in metadata and metadata['raw_llm'] is not None:
        (solution_dir / 'llm_response.txt').write_text(metadata['raw_llm'], encoding='utf-8')
    return coding_path

@app.get("/health")
def health():
    return {"note": ASCII_ART, "status": "ok", "model_configured": bool(GOOGLE_API_KEY)}

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    """

    Upload ZIP (statement, sample_in, sample_out). LLM generates code until sample matches expected.
    Returns JSON with 'solution' text and 'solution_id' and saves artifacts server-side.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY env var for generation.")

    tmp_root = Path(tempfile.mkdtemp(prefix="gen_"))
    try:
        content = await file.read()
        unpack_zip_to_dir(content, tmp_root)
        found = find_problem_files(tmp_root)
        if 'statement' not in found or 'sample_in' not in found or 'sample_out' not in found:
            raise HTTPException(status_code=400, detail="Zip must contain statement, sample_in, sample_out files.")

        statement_text = found['statement'].read_text(encoding='utf-8')
        sample_in_text = found['sample_in'].read_text(encoding='utf-8')
        sample_out_text = found['sample_out'].read_text(encoding='utf-8')

        # import LLM client
        try:
            import google.generativeai as genai
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        last_code = None
        last_error = ""
        raw_llm_text = None
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            if attempt == 1:
                prompt = f"""You are an expert competitive programmer. Write a Python 3 solution that reads from standard input and writes to standard output.

Problem statement:
{statement_text}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}
Keep in mind the constraints as mentioned in the statement and generate the most efficient and optimal code.
"""
                # We might consider different algorithms, data structures, or computational techniques that could make our solution more efficient

            else:
                prompt = f"""Previous submission produced incorrect output or runtime errors.

Problem statement:
{statement_text}

Previous code:
```python
{last_code}
```

Reason for failure (most recent):
{last_error}

Sample Input:
{sample_in_text}

Expected Sample Output:
{sample_out_text}

Keep in mind the constraints as mentioned in the statement and generate the most efficient and optimal code.
"""
            try:
                resp = model.generate_content(prompt)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

            raw_llm_text = resp.text or ""
            code_candidate = extract_python_from_markdown(raw_llm_text) or (resp.text or "").strip()
            last_code = code_candidate

            # Run candidate on sample input
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

            # Decide outcome
            if sample_run["timed_out"]:
                last_error = f"Sample run timed out: {sample_run['stderr']}"
                continue
            if sample_run["stderr"]:
                last_error = f"Sample runtime error: {sample_run['stderr']}"
                continue
            if sample_out_norm != expected_norm:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
                continue

            # Success -> persist and return
            solution_id = str(uuid.uuid4())
            solution_dir = SOLUTIONS_DIR / solution_id
            metadata = {"solution_id": solution_id, "attempt": attempt, "raw_llm": raw_llm_text}
            coding_path = save_solution_on_server(code_candidate, solution_dir, metadata)

            # save other artifacts
            (solution_dir / 'sample_stdout.txt').write_text(sample_run['stdout'], encoding='utf-8')
            (solution_dir / 'statement.txt').write_text(statement_text, encoding='utf-8')
            (solution_dir / 'sample_in.txt').write_text(sample_in_text, encoding='utf-8')
            (solution_dir / 'sample_out.txt').write_text(sample_out_text, encoding='utf-8')
            (solution_dir / 'gen_response.json').write_text(str({"attempt": attempt, "raw_llm": raw_llm_text}), encoding='utf-8')

            return JSONResponse({
                'status': 'generated',
                'solution_id': solution_id,
                'sample_stdout': sample_run['stdout'],
                'solution_path': str(coding_path),
                'solution': code_candidate,
                'raw_llm': raw_llm_text
            })

        # exhausted attempts
        return JSONResponse({'status': 'failed', 'attempts': MAX_GENERATION_ATTEMPTS, 'last_error': last_error, 'last_solution': last_code}, status_code=400)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

@app.post("/test")
async def test_solution(
    solution_id: Optional[str] = Form(None),
    test_input: Optional[str] = Form(None),
    test_file: Optional[UploadFile] = File(None),
    solution: Optional[str] = Form(None),
    solution_file: Optional[UploadFile] = File(None),
    problem_zip: Optional[UploadFile] = File(None),
    statement: Optional[str] = Form(None),
    sample_in: Optional[str] = Form(None),
    sample_out: Optional[str] = Form(None),
    test_expected: Optional[str] = Form(None)
):
    """
    Multiple invocation modes for testing:
      - solution_id + test_file/test_input
      - solution_file + problem_zip + test_file
      - solution (text) + test_file (and optional statement/sample fields)
    If initial run fails, will attempt LLM-driven regeneration up to MAX_GENERATION_ATTEMPTS.
    """
    # Validate test input
    if test_file is None and test_input is None:
        raise HTTPException(status_code=400, detail="Provide test_file (upload) or test_input (form field).")

    # Determine solution to use / create or overwrite as requested
    created_new = False
    if solution_file is not None or solution is not None or problem_zip is not None:
        # Caller provided code or problem zip; create or reuse solution_id
        if solution_id is None:
            solution_id = str(uuid.uuid4())
            created_new = True
        solution_dir = SOLUTIONS_DIR / solution_id
        solution_dir.mkdir(parents=True, exist_ok=True)

        # If problem_zip provided, extract statement/sample files
        if problem_zip is not None:
            zip_bytes = await problem_zip.read()
            tempd = Path(tempfile.mkdtemp())
            try:
                unpack_zip_to_dir(zip_bytes, tempd)
                found = find_problem_files(tempd)
                if 'statement' in found:
                    shutil.copy(found['statement'], solution_dir / 'statement.txt')
                if 'sample_in' in found:
                    shutil.copy(found['sample_in'], solution_dir / 'sample_in.txt')
                if 'sample_out' in found:
                    shutil.copy(found['sample_out'], solution_dir / 'sample_out.txt')
            finally:
                try:
                    shutil.rmtree(tempd)
                except Exception:
                    pass

        # If solution_file uploaded, save it. If solution text provided, save that.
        if solution_file is not None:
            sol_bytes = await solution_file.read()
            sol_text = sol_bytes.decode('utf-8', errors='replace')
            sol_text = extract_python_from_markdown(sol_text) or sol_text.strip()
            metadata = {"solution_id": solution_id, "provided_file": True}
            coding_path = save_solution_on_server(sol_text, solution_dir, metadata)
        elif solution is not None:
            sol_text = extract_python_from_markdown(solution) or solution.strip()
            metadata = {"solution_id": solution_id, "provided_directly": True}
            coding_path = save_solution_on_server(sol_text, solution_dir, metadata)
        else:
            coding_path = solution_dir / 'coding_solution.py'
            if not coding_path.exists():
                raise HTTPException(status_code=400, detail="No solution provided and no existing coding_solution.py for this solution_id")
    else:
        # No new solution provided - use existing solution_id
        if solution_id is None:
            raise HTTPException(status_code=400, detail="Provide either an existing solution_id or upload solution_file/solution text")
        solution_dir = SOLUTIONS_DIR / solution_id
        if not solution_dir.exists():
            raise HTTPException(status_code=404, detail="solution_id not found")
        coding_path = solution_dir / 'coding_solution.py'
        if not coding_path.exists():
            raise HTTPException(status_code=404, detail="coding_solution.py not found for this solution_id")

# Read test input
    if test_file is not None:
        test_input_text = (await test_file.read()).decode('utf-8')
    else:
        test_input_text = test_input

    # Run current code on test input
    current_code = coding_path.read_text(encoding='utf-8')
    run_res = run_python_code_str(current_code, test_input_text, timeout=EXECUTION_TIMEOUT)

    def normalize_out(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines())

    # If ok and matches expected (if provided), save and return
    if not run_res['timed_out'] and run_res['stderr'] == '' and (test_expected is None or normalize_out(run_res['stdout']) == normalize_out(test_expected)):
        (solution_dir / 'test_output.txt').write_text(run_res['stdout'], encoding='utf-8')
        return JSONResponse({
            'status': 'ok',
            'solution_id': solution_id,
            'test_stdout': run_res['stdout'],
            'test_stderr': run_res['stderr'],
            'test_output_path': str(solution_dir / 'test_output.txt'),
            'solution': current_code
        })

    # Save raw output for debugging
    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'], encoding='utf-8')

    # If no LLM available, return failure with run result
    if not GOOGLE_API_KEY:
        return JSONResponse({'status': 'failed', 'reason': 'no_google_api_key', 'run_result': run_res, 'solution': current_code}, status_code=400)

    # Prepare regeneration loop - use LLM to fix code
    try:
        import google.generativeai as genai
    except Exception as e:
        return JSONResponse({'status': 'failed', 'reason': f'missing_llm_lib: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    last_code = current_code
    last_error = f"Initial run failed. stdout:\n{run_res['stdout']}\nstderr:\n{run_res['stderr']}"

    # Use available statement/sample context if present
    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else (statement or "")
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else (sample_in or "")
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else (sample_out or "")

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed on a later test input. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{statement_text}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

Previous code:
```python
{last_code}
```

Failure when running on this test input:
Test Input:
{test_input_text}

Failure details:
{last_error}

If a corrected solution is provided, reply with the full Python code in a single markdown code block (```python ... ```).
"""
        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            return JSONResponse({'status': 'failed', 'reason': f'LLM_call_failed: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)

        raw_llm_text = resp.text or ""
        code_candidate = extract_python_from_markdown(raw_llm_text) or (raw_llm_text or "").strip()
        last_code = code_candidate

        # Ensure candidate still passes sample tests if samples are available
        if sample_in_text and sample_out_text:
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

            if sample_run["timed_out"]:
                last_error = f"Sample run timed out: {sample_run['stderr']}"
                continue
            if sample_run["stderr"]:
                last_error = f"Sample runtime error after regen: {sample_run['stderr']}"
                continue
            if sample_out_norm != expected_norm:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch after regen. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
                continue

        # Run candidate on test input
        test_run = run_python_code_str(code_candidate, test_input_text, timeout=EXECUTION_TIMEOUT)
        test_out_norm = "\n".join(line.rstrip() for line in test_run["stdout"].strip().splitlines())

        if test_run["timed_out"]:
            last_error = f"Test run timed out: {test_run['stderr']}"
            continue
        if test_run["stderr"]:
            last_error = f"Test runtime error after regen: {test_run['stderr']}"
            continue
        if test_expected is not None:
            expected_test_norm = "\n".join(line.rstrip() for line in test_expected.strip().splitlines())
            if test_out_norm != expected_test_norm:
                diff = "".join(difflib.unified_diff(expected_test_norm.splitlines(keepends=True), test_out_norm.splitlines(keepends=True), fromfile='expected_test', tofile='actual_test'))
                last_error = f"Test mismatch after regen. Diff:\n{diff}\nStdout:\n{test_run['stdout']}\nStderr:\n{test_run['stderr']}"
                continue
        else:
            if test_out_norm == "":
                last_error = "Test run produced empty stdout after regen."
                continue

        # Success: overwrite saved solution and write test output & llm response
        coding_path.write_text(code_candidate, encoding='utf-8')
        (solution_dir / 'test_output.txt').write_text(test_run['stdout'], encoding='utf-8')
        (solution_dir / 'llm_response.txt').write_text(raw_llm_text, encoding='utf-8')
        (solution_dir / 'metadata.json').write_text( str({"solution_id": solution_id, "regenerated_attempt": attempt}),encoding='utf-8')
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': test_run['stdout'], 'test_stderr': test_run['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'solution': code_candidate, 'attempts': attempt})

    # exhausted regeneration
    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'], encoding='utf-8')
    return JSONResponse({'status': 'failed', 'reason': 'regeneration_exhausted', 'last_error': last_error, 'last_solution': last_code, 'solution': last_code}, status_code=400)

@app.get("/download/{solution_id}/{filename}")
def download_file(solution_id: str, filename: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail='solution_id not found')
    if filename not in ALLOWED_DOWNLOAD_FILES:
        raise HTTPException(status_code=400, detail='Requested filename not allowed')
    target = solution_dir / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(str(target), media_type='application/octet-stream', filename=filename)

@app.get("/solutions/{solution_id}/files")
def list_solution_files(solution_id: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail='solution_id not found')
    files = [p.name for p in solution_dir.iterdir() if p.is_file()]
    return {'solution_id': solution_id, 'files': files}

@app.get("/solutions")
def list_solutions():
    ids = []
    for p in SOLUTIONS_DIR.iterdir():
        if p.is_dir():
            ids.append({'solution_id': p.name, 'files': [f.name for f in p.iterdir() if f.is_file()]})
    return {'solutions': ids}






# ------------------------------------------------------------



### Chunk 4 — `/test2` (JSON client-run) — paste this after chunk 3
# ```python
# ---- New endpoint: POST /test2 (JSON) for client-local-run workflow ----
@app.post("/test2")
async def test2_endpoint_json(payload: Dict[str, Any] = Body(...)):
    """
    Accept JSON from local runner. Fields:
      - solution_id (optional if 'solution' provided)
      - solution (optional code text to save)
      - stdout, stderr, timed_out (bool), returncode, test_input (string), test_expected (optional)
    Behavior:
      - Save run output to solution dir
      - If successful locally -> return ok
      - If failed -> call LLM (same regeneration flow as /test) and return regenerated candidate
    """
    data = payload or {}
    solution_id = data.get("solution_id")
    provided_solution_text = data.get("solution")
    stdout = data.get("stdout", "") or ""
    stderr = data.get("stderr", "") or ""
    timed_out = bool(data.get("timed_out", False))
    returncode = data.get("returncode", None)
    test_input_text = data.get("test_input", "") or ""
    test_expected = data.get("test_expected", None)

    # validation
    if not solution_id and not provided_solution_text:
        raise HTTPException(status_code=400, detail="Either solution_id or solution (code text) must be provided")

    # create or get solution dir
    created_new = False
    if provided_solution_text and not solution_id:
        solution_id = str(uuid.uuid4())
        created_new = True
        solution_dir = SOLUTIONS_DIR / solution_id
        solution_dir.mkdir(parents=True, exist_ok=True)
        code_text = extract_python_from_markdown(provided_solution_text) or provided_solution_text
        save_solution_on_server(code_text, solution_dir, {"solution_id": solution_id, "provided_via_post": True})
    else:
        solution_dir = SOLUTIONS_DIR / solution_id
        if not solution_dir.exists():
            raise HTTPException(status_code=404, detail="solution_id not found")

    coding_path = solution_dir / "coding_solution.py"
    if not coding_path.exists() and not provided_solution_text:
        raise HTTPException(status_code=404, detail="coding_solution.py not found for this solution_id")

    # save run output
    try:
        (solution_dir / "test_output.txt").write_text(stdout + "\n[stderr]\n" + stderr, encoding="utf-8")
        if test_input_text:
            (solution_dir / "latest_test_input.txt").write_text(test_input_text, encoding="utf-8")
    except Exception:
        pass

    def _normalize_out(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines())

    # If client already reports success, return ok immediately (server won't re-run)
    if (not timed_out) and stderr.strip() == "" and (test_expected is None or _normalize_out(stdout) == _normalize_out(test_expected)):
        return JSONResponse({"status": "ok", "solution_id": solution_id, "test_stdout": stdout, "test_stderr": stderr})

    # Otherwise we need to attempt regeneration using the LLM (same logic as /test)

    last_code = coding_path.read_text(encoding="utf-8") if coding_path.exists() else (provided_solution_text or "")
    last_error = f"Local run failed. timed_out={timed_out}\nreturncode={returncode}\nstderr:\n{stderr}\nstdout:\n{stdout}"

    # sample/statement context if available
    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else ""
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else ""
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else ""

    # If no LLM available, return failure
    if not GOOGLE_API_KEY:
        (solution_dir / "metadata.json").write_text(json.dumps({"solution_id": solution_id, "last_error": last_error, "timestamp": time.time()}), encoding="utf-8")
        return JSONResponse({"status": "failed", "reason": "no_google_api_key", "last_error": last_error}, status_code=400)

    try:
        import google.generativeai as genai
    except Exception as e:
        return JSONResponse({"status": "failed", "reason": f"missing_llm_lib: {e}", "last_error": last_error}, status_code=500)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    raw_llm_text = None
    last_solution = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed when run locally. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{statement_text}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

Previous code:
{last_code}

Failure when running on this test input:
Test Input:
{test_input_text}

Failure details:
{last_error}

If a corrected solution is provided, reply with the full Python code in a single markdown code block (```python ... ```).
"""
        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            last_error = f"LLM call failed: {e}"
            continue

        raw_llm_text = getattr(resp, "text", str(resp)) or ""
        code_candidate = extract_python_from_markdown(raw_llm_text) or (raw_llm_text or "").strip()
        last_solution = code_candidate

        # validate sample tests if present
        if sample_in_text and sample_out_text:
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())
            if sample_run["timed_out"]:
                last_error = f"Sample run timed out after regen: {sample_run['stderr']}"
                continue
            if sample_run["stderr"]:
                last_error = f"Sample runtime error after regen: {sample_run['stderr']}"
                continue
            if sample_out_norm != expected_norm:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch after regen. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
                continue

        # run candidate on the failing test input posted by client (if provided)
        test_run = run_python_code_str(code_candidate, test_input_text, timeout=EXECUTION_TIMEOUT)
        test_out_norm = "\n".join(line.rstrip() for line in test_run["stdout"].strip().splitlines())

        if test_run["timed_out"]:
            last_error = f"Test run timed out: {test_run['stderr']}"
            continue
        if test_run["stderr"]:
            last_error = f"Test runtime error after regen: {test_run['stderr']}"
            continue
        if test_expected is not None:
            expected_test_norm = "\n".join(line.rstrip() for line in test_expected.strip().splitlines())
            if test_out_norm != expected_test_norm:
                diff = "".join(difflib.unified_diff(expected_test_norm.splitlines(keepends=True), test_out_norm.splitlines(keepends=True), fromfile='expected_test', tofile='actual_test'))
                last_error = f"Test mismatch after regen. Diff:\n{diff}\nStdout:\n{test_run['stdout']}\nStderr:\n{test_run['stderr']}"
                continue
        else:
            if test_out_norm == "":
                last_error = "Test run produced empty stdout after regen."
                continue

        # success - persist candidate and return regeneration response
        try:
            coding_path.write_text(code_candidate, encoding='utf-8')
            (solution_dir / 'test_output.txt').write_text(test_run['stdout'], encoding='utf-8')
            (solution_dir / 'llm_response.txt').write_text(raw_llm_text, encoding='utf-8')
            (solution_dir / 'metadata.json').write_text(json.dumps({"solution_id": solution_id, "regenerated_attempt": attempt, "timestamp": time.time()}), encoding='utf-8')
        except Exception:
            pass

        return JSONResponse({"status": "regenerated", "solution_id": solution_id, "candidate_solution": code_candidate, "raw_llm_text": raw_llm_text, "attempts": attempt})

    # exhausted all regen attempts
    return JSONResponse({"status": "failed", "reason": "regeneration_exhausted", "last_error": last_error, "last_solution": last_solution, "raw_llm_text": raw_llm_text}, status_code=400)



@app.delete("/cleanup/{solution_id}")
def cleanup_solution(solution_id: str):
    """
    Remove solution artifacts for solution_id from server storage.
    Deletes the directory SOLUTIONS_DIR/<solution_id> and its contents.
    """
    solution_dir = (SOLUTIONS_DIR / solution_id).resolve()
    base = SOLUTIONS_DIR.resolve()

    # Safety: ensure deletion target is inside SOLUTIONS_DIR
    if not str(solution_dir).startswith(str(base) + os.sep) and solution_dir != base:
        raise HTTPException(status_code=400, detail="Invalid solution_id / path")

    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")

    try:
        shutil.rmtree(solution_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove solution files: {e}")

    return JSONResponse({"status": "ok", "solution_id": solution_id, "message": "solution files removed"})



##HELPER SCRIPT

#!/usr/bin/env python3
"""
# test2_automate_nonimage_post_on_error.py

# Run a locally available generated program against a test input. If it fails (stderr, non-zero return code,
# or timeout), POST the run result to the server's non-image /test2 endpoint so the server can attempt LLM
# regeneration. If the server returns a regenerated candidate, overwrite the local program and retry (up to max-iters).

# Usage examples:

# # Use an existing solution_id on server, run local program or download it first:
# python test2_automate_nonimage_post_on_error.py \
#   --server https://my-api-mneh.onrender.com \
#   --solution-id <SOLUTION_ID> \
#   --test-file pp_input.txt \
#   --program-filename coding_solution.py

# # Upload a local program and test it locally; only post to server on failure and include local program in POST:
# python test2_automate_nonimage_post_on_error.py \
#   --server https://my-api-mneh.onrender.com \
#   --solution-file a.py \
#   --test-file pp_input.txt \
#   --upload-local-program

# # Provide test input string instead of file:
# python test2_automate_nonimage_post_on_error.py \
#   --server https://my-api-mneh.onrender.com \
#   --solution-file a.py \
#   --test-input "1 2 3\n" \
#   --upload-local-program

# """

# import argparse
# import requests
# import sys
# import shutil
# import subprocess
# import time
# from pathlib import Path

# DOWNLOAD_TIMEOUT = 60
# DEFAULT_MAX_ITERS = 4
# DEFAULT_TIMEOUT = 60

# def download_program(server, solution_id, program_filename, out_path):
#     url = server.rstrip('/') + f'/download/{solution_id}/{program_filename}'
#     r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
#     r.raise_for_status()
#     with open(out_path, 'wb') as f:
#         shutil.copyfileobj(r.raw, f)
#     return out_path

# def run_program_locally(python_exe, program_path, test_input_path, time_limit):
#     """
#     Run program_path using python_exe with test_input_path as stdin.
#     Returns dict {stdout, stderr, timed_out, returncode}
#     """
#     try:
#         with open(test_input_path, 'r', encoding='utf-8') as inf:
#             p = subprocess.run([python_exe, str(program_path)],
#                                stdin=inf, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                                text=True, timeout=time_limit)
#             return {"stdout": p.stdout or "", "stderr": p.stderr or "", "timed_out": False, "returncode": p.returncode}
#     except subprocess.TimeoutExpired:
#         return {"stdout": "", "stderr": f"Time Limit Exceeded ({time_limit} seconds)", "timed_out": True, "returncode": None}
#     except Exception as e:
#         return {"stdout": "", "stderr": f"Local runner exception: {repr(e)}", "timed_out": False, "returncode": None}

# def post_result(server, payload):
#     url = server.rstrip('/') + '/test2'
#     r = requests.post(url, json=payload, timeout=DOWNLOAD_TIMEOUT)
#     r.raise_for_status()
#     return r.json()

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--server', required=True, help='Base URL of the server (example: https://my-api-mneh.onrender.com)')
#     p.add_argument('--solution-id', help='Existing solution_id on the server (optional)')
#     p.add_argument('--solution-file', help='Local solution code file to use (optional)')
#     p.add_argument('--test-file', help='Local test input file (required unless --test-input is provided)')
#     p.add_argument('--test-input', help='Direct test input string (alternative to --test-file)')
#     p.add_argument('--program-filename', default='coding_solution.py', help='Name to save or use for the program file locally')
#     p.add_argument('--python-exe', default=sys.executable, help='Python executable to run locally')
#     p.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS)
#     p.add_argument('--time-limit', type=int, default=DEFAULT_TIMEOUT)
#     p.add_argument('--upload-local-program', action='store_true', help='When posting failure to server, include the local program source in payload["solution"]')
#     args = p.parse_args()

#     if not args.test_file and args.test_input is None:
#         print("Either --test-file or --test-input must be provided.", file=sys.stderr)
#         sys.exit(2)

#     # Acquire program locally
#     local_program_path = None
#     if args.solution_file:
#         local_program_path = Path(args.solution_file)
#         if not local_program_path.exists():
#             print("solution_file not found:", local_program_path, file=sys.stderr)
#             sys.exit(2)
#     elif args.solution_id:
#         # try to download the program from server
#         try:
#             print("Downloading program from server...")
#             download_program(args.server, args.solution_id, args.program_filename, Path(args.program_filename))
#             local_program_path = Path(args.program_filename)
#             print("Downloaded program to", local_program_path)
#         except Exception as e:
#             print("Failed to download program:", e, file=sys.stderr)
#             print("Provide --solution-file if you have a local copy.", file=sys.stderr)
#             sys.exit(1)
#     else:
#         print("Provide either --solution-file or --solution-id to obtain a program to run.", file=sys.stderr)
#         sys.exit(2)

#     # Prepare test input file locally
#     if args.test_file:
#         local_test_input = Path(args.test_file)
#         if not local_test_input.exists():
#             print("test file not found:", local_test_input, file=sys.stderr)
#             sys.exit(2)
#     else:
#         local_test_input = Path('test_input_temp.txt')
#         local_test_input.write_text(args.test_input or '', encoding='utf-8')

#     python_exe = args.python_exe
#     max_iters = args.max_iters
#     time_limit = args.time_limit

#     for attempt in range(1, max_iters + 1):
#         print(f"\n=== ITERATION {attempt}/{max_iters} ===")
#         run_res = run_program_locally(python_exe, local_program_path, local_test_input, time_limit)
#         stdout = run_res['stdout']
#         stderr = run_res['stderr']
#         timed_out = run_res['timed_out']
#         returncode = run_res['returncode']

#         # Save latest local run logs for debug
#         Path('latest_run_stdout.txt').write_text(stdout or '', encoding='utf-8')
#         Path('latest_run_stderr.txt').write_text(stderr or '', encoding='utf-8')

#         # Determine local success: no stderr, not timed out, and returncode==0 (or None treated as success only if no stderr)
#         success = (not timed_out) and (not stderr.strip()) and (returncode == 0 or returncode is None)
#         if success:
#             print("✅ Local run successful. No server contact required. Exiting.")
#             sys.exit(0)

#         # Otherwise, prepare payload and POST to server /test2 for regeneration
#         payload = {
#             'solution_id': args.solution_id,
#             'stdout': stdout or '',
#             'stderr': stderr or '',
#             'timed_out': bool(timed_out),
#             'returncode': returncode,
#             'test_input': local_test_input.read_text(encoding='utf-8')
#         }
#         if args.upload_local_program:
#             payload['solution'] = local_program_path.read_text(encoding='utf-8')

#         print("Posting failure to server /test2 for regeneration...")
#         try:
#             server_resp = post_result(args.server, payload)
#         except Exception as e:
#             print("Failed to post to server:", e, file=sys.stderr)
#             sys.exit(1)

#         # Interpret server response
#         status = server_resp.get('status')
#         if status == 'regenerated':
#             # Server provided a candidate solution to try locally
#             candidate = server_resp.get('candidate_solution') or server_resp.get('solution') or server_resp.get('candidate') or server_resp.get('raw_llm_text')
#             if not candidate:
#                 print("Server declared regeneration but did not return candidate code. Response:", server_resp, file=sys.stderr)
#                 sys.exit(1)
#             print("Received regenerated candidate from server. Overwriting local program and retrying...")
#             local_program_path.write_text(candidate, encoding='utf-8')
#             # If server returned a new solution_id, update it
#             if 'solution_id' in server_resp:
#                 args.solution_id = server_resp['solution_id']
#             # loop will rerun
#             continue
#         elif status == 'ok':
#             print("Server responded OK (candidate verified server-side). Exiting.")
#             sys.exit(0)
#         else:
#             print("Server returned failure or unexpected response. Response:", server_resp, file=sys.stderr)
#             sys.exit(1)

#     print("Reached maximum iterations without success.")
#     sys.exit(2)

# if __name__ == "__main__":
#     main()
