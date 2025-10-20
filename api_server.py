# api_server_two_step.py
# Two-step FastAPI server:
# 1) /generate  - upload problem zip (statement, sample_in, sample_out). The server generates code until it passes the sample.
#                 On success it saves the solution as solutions/<id>/coding_solution.py and returns {"solution_id": id, ...}
# 2) /test      - provide solution_id and test_input (and optional test_expected). The server runs the saved solution on test_input.
#                 If it fails, the server asks the LLM to regenerate corrected code (using the test_input & error as feedback)
#                 and repeats until success or attempts exhausted. On success it saves test_output.txt.
#
# Security note: This runs generated code with python -c and is not fully sandboxed. For production, run executed code inside containers or stricter sandboxes.
#
# Usage: uvicorn api_server_two_step:app --host 0.0.0.0 --port $PORT

import os
import shutil
import zipfile
import tempfile
import subprocess
import difflib
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Code-Gen Two-Step API")

# Configuration
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "8"))  # seconds

def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    if "```python" in text:
        block = text.split("```python", 1)[1]
        block = block.split("```", 1)[0]
        return block.strip()
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0]
        lines = block.splitlines()
        if lines and lines[0].strip().lower().startswith("python"):
            block = "\n".join(lines[1:])
        return block.strip()
    return None

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
    try:
        p = subprocess.Popen(
            [os.getenv("PYTHON_PATH", "python"), "-c", code_str],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
    Runs the LLM generation loop until a candidate passes sample_in->sample_out.
    Saves the successful solution as solutions/<id>/coding_solution.py
    Returns JSON with solution_id and sample stdout.
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

            # Run on sample input
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
                last_error = f"Sample mismatch. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
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

            return JSONResponse({
                'status': 'generated',
                'solution_id': solution_id,
                'sample_stdout': sample_run['stdout'],
                'solution_path': str(coding_path)
            })

        # exhausted attempts
        return JSONResponse({'status': 'failed', 'attempts': MAX_GENERATION_ATTEMPTS, 'last_error': last_error, 'last_solution': last_code}, status_code=400)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

@app.post("/test")
async def test_solution(solution_id: str = Form(...), test_input: str = Form(...), test_expected: Optional[str] = Form(None)):
    """
    Run the saved solution on provided test_input. If it fails (runtime error, timeout, or mismatch with test_expected),
    attempt to regenerate corrected code with the LLM using the previous code and failure trace. On success save test_output.txt
    and updated coding_solution.py.
    """
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")

    coding_path = solution_dir / 'coding_solution.py'
    if not coding_path.exists():
        raise HTTPException(status_code=404, detail="Saved solution not found for this solution_id")

    # Load supporting files (statement + samples) to include in regeneration prompts if needed
    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else ''
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else ''
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else ''

    # Read the currently saved solution and run it
    current_code = coding_path.read_text(encoding='utf-8')

    # Helper to persist test output
    def save_test_output(text: str):
        (solution_dir / 'test_output.txt').write_text(text, encoding='utf-8')

    # Run current solution on provided test_input
    run_res = run_python_code_str(current_code, test_input, timeout=EXECUTION_TIMEOUT)
    if not run_res['timed_out'] and run_res['stderr'] == '' and (test_expected is None or "\n".join(line.rstrip() for line in run_res['stdout'].strip().splitlines()) == "\n".join(line.rstrip() for line in test_expected.strip().splitlines())):
        # success: save and return
        save_test_output(run_res['stdout'])
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': run_res['stdout'], 'test_stderr': run_res['stderr']})

    # Otherwise we need to attempt regeneration using LLM feedback
    if not GOOGLE_API_KEY:
        # cannot regenerate without key; return the run result for debugging
        save_test_output(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'])
        return JSONResponse({'status': 'failed', 'reason': 'no_google_api_key', 'run_result': run_res}, status_code=400)

    # Prepare regeneration loop
    try:
        import google.generativeai as genai
    except Exception as e:
        save_test_output(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'])
        raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    last_code = current_code
    last_error = f"Initial run failed. stdout:\n{run_res['stdout']}\nstderr:\n{run_res['stderr']}"
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        # Build prompt describing previous code and failure, include test_input and (if available) expected
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
{test_input}

Failure details:
{last_error}

If a corrected solution is provided, reply with the full Python code in a single markdown code block (```python ... ```).
"""
        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            return JSONResponse({'status': 'failed', 'reason': f'LLM_call_failed: {e}'}, status_code=500)

        code_candidate = extract_python_from_markdown(resp.text or "") or (resp.text or "").strip()
        last_code = code_candidate

        # Re-run candidate on sample first
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

        # If sample passes, run on test_input
        test_run = run_python_code_str(code_candidate, test_input, timeout=EXECUTION_TIMEOUT)
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
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': test_run['stdout'], 'test_stderr': test_run['stderr'], 'attempts': attempt})

    # Exhausted regeneration attempts: save last run output and return failure
    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'], encoding='utf-8')
    return JSONResponse({'status': 'failed', 'reason': 'regeneration_exhausted', 'last_error': last_error, 'last_solution': last_code}, status_code=400)