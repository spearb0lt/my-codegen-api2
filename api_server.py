# api_server_final.py
"""
Robust Code-Gen Two-Step FastAPI server (final merge of v3+v4).
Features:
 - /health
 - /generate : upload problem ZIP (statement, sample_in, sample_out) -> LLM generates code until sample passes.
 - /test     : multiple modes including regeneration with LLM when test input fails.
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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

import re
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\\n([\\s\\S]*?)```", re.IGNORECASE)

app = FastAPI(title="Code-Gen Two-Step Final API")

# Configuration via environment
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "8"))  # seconds

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
        if ("\n" in text or "\t" in text or '\"' in text) and text.count("\n") > text.count("\n"):
            decoded = bytes(text, "utf-8").decode("unicode_escape")
            # prefer decoded if it yields more real newlines
            if decoded.count("\n") >= text.count("\n"):
                text = decoded
    except Exception:
        # fallback: leave text as-is
        pass

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
    return {"status": "ok", "model_configured": bool(GOOGLE_API_KEY)}

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

Provide only the final Python code in a single markdown code block (```python ... ```).
"""
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

Please provide a corrected complete Python solution in one markdown block.
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
                last_error = f"Sample mismatch. Diff:\\n{diff}\\nStdout:\\n{sample_run['stdout']}\\nStderr:\\n{sample_run['stderr']}"
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
    last_error = f"Initial run failed. stdout:\\n{run_res['stdout']}\\nstderr:\\n{run_res['stderr']}"

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
                last_error = f"Sample mismatch after regen. Diff:\\n{diff}\\nStdout:\\n{sample_run['stdout']}\\nStderr:\\n{sample_run['stderr']}"
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
                last_error = f"Test mismatch after regen. Diff:\\n{diff}\\nStdout:\\n{test_run['stdout']}\\nStderr:\\n{test_run['stderr']}"
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
