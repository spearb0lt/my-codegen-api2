# api_server_two_step_v2.py
"""
Two-step FastAPI server (v2) with download endpoints and support for test_input file upload.

Endpoints:
  - GET /health
  - POST /generate    : upload ZIP (statement, sample_in, sample_out). Generates code until sample passes.
                         Returns JSON including 'solution' (code text), 'solution_id', 'solution_path', 'sample_stdout'.
  - POST /test        : provide solution_id and either test_input (string form field) or test_file (uploaded file).
                         Runs saved solution on test input; if it fails and GOOGLE_API_KEY available, attempts regeneration.
                         Saves test_output.txt on server and returns JSON including 'test_stdout', 'test_stderr', 'test_output_path'.
  - GET /download/{solution_id}/{filename} : download saved files like coding_solution.py or test_output.txt
  - GET /solutions/{solution_id}/files : list saved files for solution

Security note: This service executes generated Python code via subprocess (python -c). This is NOT secure for arbitrary untrusted input.
For production, run executed code in isolated containers with no network and strict resource limits.
"""

import os
import shutil
import zipfile
import tempfile
import subprocess
import difflib
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

# Regex helper for extracting code fences
import re
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.IGNORECASE)

app = FastAPI(title="Code-Gen Two-Step API v2")

# Configuration (tweak via environment variables)
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "8"))  # seconds


def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    if "```python" in text or "```" in text:
        m = CODE_FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
    # fallback: return full text
    return text.strip()

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
        tmpf_path.unlink()
    except Exception:
        pass

def find_problem_files(workdir: Path):
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

def run_python_code_str(code_str: str, input_str: str, timeout=EXECUTION_TIMEOUT):
    """
    Run code_str with `python -c` in a subprocess and return dict with stdout, stderr, timed_out.
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


@app.get("/health")
def health():
    return {"status": "ok", "model_configured": bool(GOOGLE_API_KEY)}


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    """
    Accepts a zip with statement, sample_in, sample_out.
    Runs LLM generation loop until candidate passes sample_in->sample_out.
    Saves successful solution as solutions/<id>/coding_solution.py and returns JSON including the code text.
    """
    if not GOOGLE_API_KEY:
        # Allow generation to still run locally if you want to test without LLM, but here require key.
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

        # LLM client
        try:
            import google.generativeai as genai
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        last_code = None
        last_error = ""
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

            code_candidate = extract_python_from_markdown(resp.text or "") or (resp.text or "").strip()
            last_code = code_candidate

            # Run candidate against sample input
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

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

            # Success on sample -> persist solution
            solution_id = str(uuid.uuid4())
            solution_dir = SOLUTIONS_DIR / solution_id
            solution_dir.mkdir(parents=True, exist_ok=True)
            coding_path = solution_dir / 'coding_solution.py'
            solution_text = code_candidate
            coding_path.write_text(solution_text, encoding='utf-8')
            # save sample stdout for reference
            (solution_dir / 'sample_stdout.txt').write_text(sample_run['stdout'], encoding='utf-8')
            # also save the original statement and sample files
            (solution_dir / 'statement.txt').write_text(statement_text, encoding='utf-8')
            (solution_dir / 'sample_in.txt').write_text(sample_in_text, encoding='utf-8')
            (solution_dir / 'sample_out.txt').write_text(sample_out_text, encoding='utf-8')

            # Return JSON including the generated solution text so caller can save locally
            return JSONResponse({
                'status': 'generated',
                'solution_id': solution_id,
                'sample_stdout': sample_run['stdout'],
                'solution_path': str(coding_path),
                'solution': solution_text
            })

        # exhausted attempts
        return JSONResponse({'status': 'failed', 'attempts': MAX_GENERATION_ATTEMPTS, 'last_error': last_error, 'last_solution': last_code}, status_code=400)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass


@app.post("/test")
async def test_solution(solution_id: str = Form(...), test_input: Optional[str] = Form(None), test_file: Optional[UploadFile] = File(None), test_expected: Optional[str] = Form(None)):
    """
    Run the saved solution on provided test_input. Accepts either 'test_input' (string) or 'test_file' (uploaded file).
    If it fails, attempt regeneration using the LLM (if GOOGLE_API_KEY is set).
    On success, save solutions/<id>/test_output.txt and return JSON including test_stdout and test_output_path.
    """
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")

    coding_path = solution_dir / 'coding_solution.py'
    if not coding_path.exists():
        raise HTTPException(status_code=404, detail="Saved solution not found for this solution_id")

    # Determine test input (prefer uploaded file)
    if test_file is not None:
        test_input_text = (await test_file.read()).decode('utf-8')
    elif test_input is not None:
        test_input_text = test_input
    else:
        raise HTTPException(status_code=400, detail="Provide either 'test_input' (form) or 'test_file' (uploaded file)")

    # Helper to save test output
    def save_test_output(text: str):
        (solution_dir / 'test_output.txt').write_text(text, encoding='utf-8')

    # Load current code
    current_code = coding_path.read_text(encoding='utf-8')

    # Run current code on test input
    run_res = run_python_code_str(current_code, test_input_text, timeout=EXECUTION_TIMEOUT)
    # Quick success check: no stderr, no timeout, and if expected provided, must match
    def outputs_match(out: str, expected: str) -> bool:
        return "\n".join(line.rstrip() for line in out.strip().splitlines()) == "\n".join(line.rstrip() for line in expected.strip().splitlines())

    if not run_res['timed_out'] and run_res['stderr'] == '' and (test_expected is None or outputs_match(run_res['stdout'], test_expected)):
        save_test_output(run_res['stdout'])
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': run_res['stdout'], 'test_stderr': run_res['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt')})

    # If we get here, initial run failed. Save raw output for debugging.
    save_test_output(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'])

    # If no GOOGLE_API_KEY available, return failure and the run result.
    if not GOOGLE_API_KEY:
        return JSONResponse({'status': 'failed', 'reason': 'no_google_api_key', 'run_result': run_res}, status_code=400)

    # Regeneration loop: try to fix using LLM, up to MAX_GENERATION_ATTEMPTS
    try:
        import google.generativeai as genai
    except Exception as e:
        return JSONResponse({'status': 'failed', 'reason': f'missing_llm_lib: {e}', 'run_result': run_res}, status_code=500)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    last_code = current_code
    last_error = f"Initial run failed. stdout:\n{run_res['stdout']}\nstderr:\n{run_res['stderr']}"

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed on a later test input. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{(solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else ''}

Sample Input:
{(solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else ''}

Sample Output:
{(solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else ''}

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
            return JSONResponse({'status': 'failed', 'reason': f'LLM_call_failed: {e}', 'run_result': run_res}, status_code=500)

        code_candidate = extract_python_from_markdown(resp.text or "") or (resp.text or "").strip()
        last_code = code_candidate

        # Re-run candidate on sample first to ensure it still passes sample tests
        sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else ''
        sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else ''
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

        # If sample passes, run on test input
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

        # Success: overwrite saved solution and write test output
        coding_path.write_text(code_candidate, encoding='utf-8')
        (solution_dir / 'test_output.txt').write_text(test_run['stdout'], encoding='utf-8')
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': test_run['stdout'], 'test_stderr': test_run['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'attempts': attempt})

    # Exhausted regeneration attempts: save last run output and return failure
    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'], encoding='utf-8')
    return JSONResponse({'status': 'failed', 'reason': 'regeneration_exhausted', 'last_error': last_error, 'last_solution': last_code}, status_code=400)


@app.get("/download/{solution_id}/{filename}")
def download_file(solution_id: str, filename: str):
    """
    Download a saved file for a solution. Allowed filenames are limited for safety.
    """
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")

    # For safety, only allow certain filenames or files inside the dir
    allowed = {'coding_solution.py', 'test_output.txt', 'sample_stdout.txt', 'statement.txt', 'sample_in.txt', 'sample_out.txt'}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Requested filename not allowed")

    target = solution_dir / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(target), media_type='application/octet-stream', filename=filename)


@app.get("/solutions/{solution_id}/files")
def list_solution_files(solution_id: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")
    files = [p.name for p in solution_dir.iterdir() if p.is_file()]
    return {"solution_id": solution_id, "files": files}
