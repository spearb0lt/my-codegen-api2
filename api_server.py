# api_server_final_multimodal.py
"""
Fully-featured Code-Gen API server (multimodal updated).
Saves artifacts, generate/test/regenerate, and can pass PIL.Image objects to Gemini (preferred method).
Run: uvicorn api_server_final_multimodal:app --host 0.0.0.0 --port 8000
"""
import json
import os, re, html, uuid, shutil, tempfile, zipfile, difflib, subprocess, mimetypes, io
from pathlib import Path
from fastapi import Request, UploadFile, File, Form
from fastapi import HTTPException
from fastapi import status
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import traceback
import time, io
from typing import Tuple, Optional
import requests as _requests_if_available
try:
    from PIL import Image as PIL_Image_for_test2
except Exception:
    PIL_Image_for_test2 = None
# import requests
# from PIL import Image
# optional libs
try:
    import requests
except Exception:
    requests = None
try:
    from PIL import Image
except Exception:
    Image = None

# Configuration
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "300"))
# MULTIMODAL_MODE = os.getenv("MULTIMODAL_MODE", "1") == "1"   # enable the PIL-image-in-prompt approach
ENABLE_IMAGE_DOWNLOAD = os.getenv("ENABLE_IMAGE_DOWNLOAD", "1") == "1"
MAX_IMAGES_IN_PROMPT = int(os.getenv("MAX_IMAGES_IN_PROMPT", "6"))
MULTIMODAL_MODE=1

ALLOWED_DOWNLOAD_FILES = {
    'coding_solution.py', 'test_output.txt', 'sample_stdout.txt',
    'statement.txt', 'sample_in.txt', 'sample_out.txt', 'llm_response.txt',
    'metadata.json', 'gen_response.json'
}

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.IGNORECASE)
PHOTO_PLACEHOLDER_RE = re.compile(r"\{\{\s*PHOTO_ID:(\d+)(?:\|WIDTH:(\d+))?\s*\}\}")
URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)

app = FastAPI(title="Code-Gen Two-Step Final API (multimodal)")

def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        # If the LLM returned literal escape sequences like "\\n" inside the text
# (for example, JSON-escaped strings), attempt to decode unicode escapes
# and prefer the decoded text only if it increases the count of real newlines.
# This avoids double-decoding already-correct text.
        if any(seq in text for seq in ("\\n", "\\t", '\\"')):
            try:
                decoded = bytes(text, "utf-8").decode("unicode_escape")
                if decoded.count("\n") > text.count("\n"):
                    text = decoded
            except Exception:
        # If decoding fails, keep the original text
                pass
    except Exception:
        pass
    #     if "\n" in text and text.count("\n") > text.count("\n"):
    #         try:
    #             decoded = bytes(text, "utf-8").decode("unicode_escape")
    #             if decoded.count("\n") >= text.count("\n"):
    #                 text = decoded
    #         except Exception:
    #             pass
    # except Exception:
    #     pass
    m = CODE_FENCE_RE.search(text)
    code = m.group(1) if m else text
    code = html.unescape(code)
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

def find_problem_files(workdir: Path) -> Dict[str, Any]:
    file_keys = {
        'statement': ['statement', 'problem'],
        'sample_in': ['sample_in', 'sample.in', 'input', 'sample-input'],
        'sample_out': ['sample_out', 'sample.out', 'output', 'sample-output']
    }
    found: Dict[str, Any] = {}
    images: List[Path] = []
    for p in workdir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        for key, patterns in file_keys.items():
            if key not in found and any(pat in name for pat in patterns):
                found[key] = p
        # heuristics for images / uri text files
        if ('image' in name or 'photo' in name) and ('uri' in name or name.endswith('.txt') or name.endswith('.url')):
            images.append(p)
            continue
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            images.append(p)
            continue
        mtype, _ = mimetypes.guess_type(str(p))
        if mtype and mtype.startswith('image/'):
            images.append(p)
    if images:
        found['images'] = images
    return found

def extract_photo_placeholders(statement: str) -> Tuple[str, List[Dict[str,str]]]:
    entries: List[Dict[str,str]] = []
    def repl(m):
        pid = m.group(1)
        width = m.group(2) or ""
        entries.append({"id": pid, "width": width})
        return f"[IMAGE id={pid}" + (f" width={width}]" if width else "]")
    new_stmt = PHOTO_PLACEHOLDER_RE.sub(repl, statement)
    return new_stmt, entries

def read_image_uri_text(p: Path) -> str:
    try:
        txt = p.read_text(encoding='utf-8', errors='replace').strip()
        return " ".join(txt.splitlines())
    except Exception:
        return ""

def fetch_image_as_pil(url: str, timeout: int = 10, max_tries: int = 3) -> Tuple[Optional["Image.Image"], str]:
    """
    Try to fetch an image URL and return (PIL.Image, debug_msg).
    Returns (None, reason) on failure.
    """
    if requests is None or Image is None:
        return None, "requests or PIL not installed on server"

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36"),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.facebook.com/",
    }

    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            with requests.Session() as s:
                # disable environment proxies if that helps: s.trust_env = False
                s.headers.update(headers)
                resp = s.get(url, timeout=timeout, allow_redirects=True, stream=True)
                status = resp.status_code
                if status != 200:
                    last_exc = f"HTTP {status}"
                    time.sleep(0.5 * attempt)
                    continue
                ctype = resp.headers.get("Content-Type", "")
                content = resp.content
                if not content:
                    last_exc = "empty response content"
                    time.sleep(0.5 * attempt)
                    continue
                try:
                    img = Image.open(io.BytesIO(content))
                    img.load()  # validate
                    debug = "ok" if ctype.startswith("image/") else f"ok (warning, content-type={ctype})"
                    return img, debug
                except Exception as e_img:
                    last_exc = f"PIL decode failed: {e_img}"
                    time.sleep(0.5 * attempt)
                    continue
        except Exception as e:
            last_exc = f"requests error: {repr(e)}"
            time.sleep(0.5 * attempt)
            continue
    return None, f"fetch_image_as_pil returned None for URL: {url} - last_exc: {last_exc}"


def run_python_code_str(code_str: str, input_str: str, timeout: int = EXECUTION_TIMEOUT) -> Dict[str, Any]:
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
    solution_dir.mkdir(parents=True, exist_ok=True)
    coding_path = solution_dir / 'coding_solution.py'
    coding_path.write_text(solution_text.replace("\r\n", "\n"), encoding='utf-8')
    (solution_dir / 'metadata.json').write_text(str(metadata), encoding='utf-8')
    if 'raw_llm' in metadata and metadata['raw_llm'] is not None:
        (solution_dir / 'llm_response.txt').write_text(metadata['raw_llm'], encoding='utf-8')
    return coding_path


def download_image_from_url(url: str, dest: Path, timeout: int = 10, max_bytes: int = 10_000_000) -> bool:
    """Download url -> dest. Returns True on success, False otherwise."""
    if requests is None:
        return False
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return False
            ct = r.headers.get("Content-Type","")
            if not ct.startswith("image/"):
                # reject non-image content types
                return False
            # stream to disk with byte limit
            written = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        return False
                    f.write(chunk)
            return True
    except Exception:
        return False


def match_images_to_placeholders(statement_text: str, found_images: list) -> tuple:
    """
    Returns (ordered_image_paths, leftover_image_paths, placeholder_entries)
    - ordered_image_paths: list of Path objects matched to placeholders in appearance order
    - leftover_image_paths: list of remaining Path objects not matched
    - placeholder_entries: list of dicts [{"id": pid, "width": width}, ...] in appearance order
    """
    # extract ordered placeholders (duplicates preserved)
    placeholder_entries = []
    for m in PHOTO_PLACEHOLDER_RE.finditer(statement_text):
        placeholder_entries.append({"id": m.group(1), "width": (m.group(2) or "")})
    ordered = []
    used = set()

    # helper to try to find file by pid
    def find_by_pid(pid):
        for p in found_images:
            if p in used:
                continue
            # match in filename
            if pid in p.name:
                return p
            # if a .txt/uri, check contents
            if p.suffix.lower() in ['.txt', '.uri', '.url', '.link']:
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                    if pid in txt:
                        return p
                except Exception:
                    pass
        return None

    # match placeholders, in order
    for ph in placeholder_entries:
        pid = ph['id']
        match = find_by_pid(pid)
        if match:
            ordered.append(match)
            used.add(match)
        else:
            # no exact match, leave a placeholder gap (we will fill later with leftovers)
            ordered.append(None)

    # leftover images (not matched)
    leftovers = [p for p in found_images if p not in used]

    # fill None slots with leftover images in natural order
    result_order = []
    li = iter(leftovers)
    for slot in ordered:
        if slot is None:
            try:
                result_order.append(next(li))
            except StopIteration:
                result_order.append(None)
        else:
            result_order.append(slot)

    # finally append any remaining leftovers after placeholder slots
    for p in li:
        result_order.append(p)

    # filter out trailing None entries
    final_order = [p for p in result_order if p is not None]
    remaining = [p for p in leftovers if p not in final_order]

    return final_order, remaining, placeholder_entries
#---------------------------------------------------------------------------------------



def _read_image_uri_text_for_test2(p: Path) -> str:
    try:
        txt = p.read_text(encoding='utf-8', errors='replace').strip()
        if not txt:
            return ""
        for ln in txt.splitlines():
            s = ln.strip()
            if s:
                return s
        return ""
    except Exception:
        return ""

def _download_image_from_url_for_test2(url: str, dest: Path, timeout: int = 10, max_bytes: int = 12_000_000) -> bool:
    if _requests_if_available is None:
        return False
    try:
        with _requests_if_available.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        try:
                            f.close()
                        except Exception:
                            pass
                        try:
                            dest.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return False
                    f.write(chunk)
            return True
    except Exception:
        return False









#---------------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_configured": bool(GOOGLE_API_KEY),
        "multimodal_mode": MULTIMODAL_MODE,
        "requests_installed": requests is not None,
        "PIL_installed": Image is not None
    }



# @app.post("/debug/fetch_image")
# async def debug_fetch_image(url: str = Form(...)):
#     img, debug = fetch_image_as_pil(url)
#     urls="https://scontent.fblr22-2.fna.fbcdn.net/v/t39.32972-6/558332054_798677149577400_6895383932844571805_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=771dbb&_nc_ohc=HDsJb-7eI8gQ7kNvwFaf4pq&_nc_oc=Adnc3LY7rWHUonm5plm-8Dlp-9StKKaJk96bwCwl620QwzXXzDzA1h1w6qfxssF_R1A&_nc_zt=14&_nc_ht=scontent.fblr22-2.fna&_nc_gid=tWrhtfRrRmGZ5fokUFs6Cw&oh=00_AffFeK5ZsuiSf0nin4mszF1FQ-KOq1wkaMj7SArz1611jA&oe=6907F05B"
#     response = requests.get(urls)
#     zz=""
#     if response.status_code == 200:
#         zz="LOL"
#     else:
#         zz="KAKA"
    
#     return {"url": url, "success": img is not None, "debug": debug, "mystat": zz}
#     # return {"url": urls, "mystat": zz}









@app.post("/generate")
# async def generate(file: UploadFile = File(...)):
async def generate(file: UploadFile = File(...), check: bool = Form(True)):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY env var for generation.")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    tmp_root = Path(tempfile.mkdtemp(prefix="gen_"))
    try:
        content = await file.read()
        unpack_zip_to_dir(content, tmp_root)
        found = find_problem_files(tmp_root)
        if 'statement' not in found or 'sample_in' not in found or 'sample_out' not in found:
            raise HTTPException(status_code=400, detail="Zip must contain statement, sample_in, sample_out files.")

        statement_text = found['statement'].read_text(encoding='utf-8')
        statement_text_replaced, photo_placeholders = extract_photo_placeholders(statement_text)

        sample_in_text = found['sample_in'].read_text(encoding='utf-8')
        sample_out_text = found['sample_out'].read_text(encoding='utf-8')

        # Discover image-like files (limit to MAX_IMAGES_IN_PROMPT)
        # raw_image_files = found.get('images', [])[:MAX_IMAGES_IN_PROMPT]
        # 1) discover the raw image files detected earlier
        raw_images = found.get('images', [])
        # limit to MAX_IMAGES_IN_PROMPT (but preserve matching behavior)
        raw_images = list(raw_images)[: max(len(raw_images), MAX_IMAGES_IN_PROMPT)]
        # raw_images = list(raw_images)[:MAX_IMAGES_IN_PROMPT]
        # 2) compute order that matches placeholders
        ordered_images, leftover_images, placeholder_entries = match_images_to_placeholders(statement_text, raw_images)
        
        # Combine ordered_images first (these align with placeholders), then any leftovers
        image_files = ordered_images + [p for p in leftover_images if p not in ordered_images]
        image_files = image_files[:MAX_IMAGES_IN_PROMPT]  # final cap

         # collect image files found
        # image_files = found.get('images', [])[:MAX_IMAGES_IN_PROMPT]
        
        image_urls = []
        pil_images = []   # list of dicts: {"name":..., "img":PIL.Image, "uri":..., "debug":...}
        multimodal_errors = []
        aa="Curnt "

        for p in image_files:
            # Determine if file is a URI text file by suffix
            if p.suffix.lower() in ['.txt', '.uri', '.url', '.link']:
                # read the URI (first non-empty line)
                try:
                    uri_text = p.read_text(encoding='utf-8', errors='replace').strip().splitlines()
                    uri = uri_text[0].strip() if uri_text else ""
                    if uri:
                        image_urls.append(uri)
                        # attempt to fetch PIL image if possible
                        img, debug = fetch_image_as_pil(uri)
                        if img is not None:
                            pil_images.append({"name": p.name, "img": img, "uri": uri, "debug": debug})
                        else:
                            multimodal_errors.append(debug)
                    else:
                        multimodal_errors.append(f"empty uri file: {p.name}")
                except Exception as e:
                    multimodal_errors.append(f"read uri file failed {p.name}: {e}")
            else:
                # local image file inside ZIP
                try:
                    # prefer to open with PIL if available
                    if Image is not None:
                        img = Image.open(str(p))
                        img.load()
                        pil_images.append({"name": p.name, "img": img, "uri": None, "debug": "ok(local)"})
                    else:
                        image_urls.append(str(p.name))
                        multimodal_errors.append(f"PIL not installed - queued local filename {p.name}")
                except Exception as e:
                    multimodal_errors.append(f"local PIL open failed for {p.name}: {e}")


        # ----- End image collection -----

        # Import and configure LLM client
        try:
            import google.genai as genai
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        last_code = None
        last_error = ""
        raw_llm_text = None
        crazy_llm_text=""
        attempt=1

        # Generation/regeneration loop
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            if attempt == 1:
                # If images are attached, you may use them to understand the sample explanation. Provide only the final Python code in a single markdown code block (```python ... ```).
#
                prompt = f"""
You are an expert competitive programmer. Write the most optimized Python 3 solution that reads from standard input and writes to standard output. 
Problem statement:
{statement_text_replaced}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

Keep in mind the constraints as mentioned in the statement and generate the most efficient and optimal code.
"""
             
            else:
                prompt = f"""
Previous submission produced incorrect output.

Problem statement:
{statement_text_replaced}

Previous code:
```python
{last_code}

Reason for failure:
{last_error}

Sample Input:
{sample_in_text}

Expected Sample Output:
{sample_out_text}

Please provide a corrected most optimized and complete Python solution in one markdown block.
"""
            resp = None
            attempt_multimodal_errors: List[str] = []

        # --- 1) Try passing PIL.Image objects as prompt parts (preferred)
            # if MULTIMODAL_MODE and pil_images:
            #     aa+=" 1 done "
            #     try:
            #         prompt_parts = [prompt] + [pi["img"] for pi in pil_images[:MAX_IMAGES_IN_PROMPT]]
            #         resp = model.generate_content(prompt_parts)
            #         raw_llm_text = getattr(resp, "text", str(resp)) or ""
            #         crazy_llm_text+= raw_llm_text
                    
            #     except Exception as e:
            #         multimodal_errors.append(f"prompt_parts failed: {repr(e)}")
            #         resp = None
            
            
            if resp is None:
                aa+=" 2 also "
                if pil_images or image_urls or photo_placeholders:
                    prompt += "\nI will also prov/ide you the images in same order as that of their occurance. Make sure gather sufficient insights from them and approach the problem."
    ## 2) Fallback: text-only prompt that includes explicit image mapping & URLs
                mapping_lines = []
                for idx, pi in enumerate(pil_images, start=1):
                    uri = pi.get("uri") or pi.get("name") or f"attachment_{idx}"
                    mapping_lines.append(f"[IMG{idx}] -> {uri}")
                # include any additional image_urls (from txt files)
                for u in image_urls:
                    if u not in [pi.get("uri") for pi in pil_images if pi.get("uri")]:
                        mapping_lines.append(f"- url: {u}")
                mapping_text = ("\n-- Image mapping (attachment order) --\n" + "\n".join(mapping_lines)) if mapping_lines else ""
                prompt_with_map = prompt + ("\n\n" + mapping_text if mapping_text else "")
                try:
                    resp = model.generate_content(prompt_with_map)
                except Exception as e:
                    # cannot proceed further (avoid passing unknown kwargs)
                    raise HTTPException(status_code=500, detail=f"LLM generate_content failed: {e}; multimodal_errors: {multimodal_errors}")





            # WORKED WELL IN FAILURE
            # If that failed or we don't have PIL images, fall back to text-only prompt that
# explicitly maps attachments and includes the image URLs inline.
            # if resp is None:
            #     # Build a deterministic mapping text (Image 1 -> PHOTO_ID:..., Image 2 -> ...)
            #     aa+=" 2 done "
            #     mapping_lines = []
            #     image_urls_for_text = []
            #     for idx, pi in enumerate(pil_images, start=1):
            #         uri = pi.get("uri") or pi.get("name") or f"attachment_{idx}"
            #         mapping_lines.append(f"[IMG{idx}] -> {uri}")
            #         if pi.get("uri") and (pi["uri"].startswith("http://") or pi["uri"].startswith("https://")):
            #             image_urls_for_text.append(pi["uri"])
            #     # Also include any image_urls collected earlier (from .txt files)
            #     for u in image_urls:
            #         if u not in image_urls_for_text:
            #             image_urls_for_text.append(u)
            
            #     if mapping_lines:
            #         prompt_with_map = prompt + "\n\n-- Image mapping (in attachment order) --\n" + "\n".join(mapping_lines)
            #     else:
            #         prompt_with_map = prompt
            #     # Append the image URLs list (explicit) so model can optionally fetch/consider them
            #     if image_urls_for_text:
            #         prompt_with_map += "\n\nImage URLs:\n" + "\n".join(image_urls_for_text)
            
            #     # Final attempt: text-only
            #     try:
            #         resp = model.generate_content(prompt_with_map)
            #     except Exception as e:
            #         multimodal_errors.append(f"text-only generate_content failed: {repr(e)}")
            #         # No more fallbacks acceptable (avoid passing unknown kwargs); raise or return an informative failure
            #         raise HTTPException(status_code=500, detail=f"LLM generate failed. multimodal_errors={multimodal_errors}")
    


            
            # if resp is None:
            #     aa+=" 4 done "
            #     image_map_text = ""
            #     if photo_placeholders:
            #         pid_map_lines = []
            #         for ph in photo_placeholders:
            #             pid = ph['id']
            #             width = ph['width']
            #             match_file = None
            #             for p in image_files:
            #                 if pid in p.name:
            #                     match_file = p.name
            #                     break
            #             pid_map_lines.append(f"- PHOTO_ID {pid} -> file {match_file or '[no match]'} (width={width})")
            #         image_map_text += "\n".join(pid_map_lines)
            #     if image_urls:
            #         image_map_text += "\nImage URLs:\n" + "\n".join(image_urls)
            #     if image_map_text:
            #         prompt = prompt + "\n\n" + image_map_text
            #     try:
            #         resp = model.generate_content(prompt)
            #     except Exception:
            #         attempt_multimodal_errors.append("final text-only generate_content failed: " + traceback.format_exc())
            #         raise HTTPException(status_code=500, detail=f"LLM generate_content failed; errors: {attempt_multimodal_errors + multimodal_errors}")
    
            # Save attempt errors into multimodal_errors for diagnostics
            # multimodal_errors.extend(attempt_multimodal_errors)
    
            raw_llm_text = getattr(resp, "text", str(resp)) or ""
            code_candidate = extract_python_from_markdown(raw_llm_text) or raw_llm_text.strip()
            last_code = code_candidate
    
            # Run candidate on sample input
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())
    
            if sample_run["timed_out"]:
                last_error = sample_run["stderr"]
                continue
            if sample_run["stderr"]:
                last_error = sample_run["stderr"]
                continue
            if ((out_norm == expected_norm) and check) or not check:
                # success -> persist artifacts
                solution_id = str(uuid.uuid4())
                solution_dir = SOLUTIONS_DIR / solution_id
                metadata = {
                    "solution_id": solution_id,
                    "attempt": attempt,
                    "raw_llm": raw_llm_text,
                    "multimodal_errors": multimodal_errors,
                    "images": [p.name for p in image_files]
                    
                    # "image_files": [p.name for p in image_files],
                    # "image_urls": image_urls
                }
                coding_path = save_solution_on_server(code_candidate, solution_dir, metadata)
                (solution_dir / 'sample_stdout.txt').write_text(sample_run['stdout'], encoding='utf-8')
                (solution_dir / 'statement.txt').write_text(statement_text, encoding='utf-8')
                (solution_dir / 'sample_in.txt').write_text(sample_in_text, encoding='utf-8')
                (solution_dir / 'sample_out.txt').write_text(sample_out_text, encoding='utf-8')
                (solution_dir / 'gen_response.json').write_text(str({"attempt": attempt}), encoding='utf-8')
                for p in image_files:
                    try:
                        shutil.copy(p, solution_dir / p.name)
                    except Exception:
                        pass
                (solution_dir / 'metadata.json').write_text(str(metadata), encoding='utf-8')
    
                return JSONResponse({
                    "status": "generated",
                    "actually": aa,
                    "attempt": attempt,
                     "crazy": crazy_llm_text,
                    "solution_id": solution_id,
                    "sample_stdout": sample_run["stdout"],
                    "solution": code_candidate,
                    "raw_llm_text": raw_llm_text,
                    "multimodal_errors": multimodal_errors,
                    "statement":prompt_with_map
                })
            else:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), out_norm.splitlines(keepends=True), fromfile="expected", tofile="actual"))
                last_error = f"Wrong output. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
    
        # exhausted attempts
        return JSONResponse({
            "status": "failed",
            "actually": aa,
            "attempts": MAX_GENERATION_ATTEMPTS,
            "crazy": crazy_llm_text,
            "multimodal_errors": multimodal_errors,
            "last_error": last_error,
            "last_solution": last_code,
            "raw_llm_text": raw_llm_text
        }, status_code=400)
    
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

@app.post('/test')
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
    test_expected: Optional[str] = Form(None),
    check: bool = Form(True)
):
    if test_file is None and test_input is None:
        raise HTTPException(status_code=400, detail='Provide test_file or test_input.')

    created_new = False
    if solution_file is not None or solution is not None or problem_zip is not None:
        if solution_id is None:
            solution_id = str(uuid.uuid4()); created_new = True
        solution_dir = SOLUTIONS_DIR / solution_id; solution_dir.mkdir(parents=True, exist_ok=True)
        if problem_zip is not None:
            zip_bytes = await problem_zip.read(); tmpd = Path(tempfile.mkdtemp());
            try:
                unpack_zip_to_dir(zip_bytes, tmpd); found = find_problem_files(tmpd)
                if 'statement' in found: shutil.copy(found['statement'], solution_dir / 'statement.txt')
                if 'sample_in' in found: shutil.copy(found['sample_in'], solution_dir / 'sample_in.txt')
                if 'sample_out' in found: shutil.copy(found['sample_out'], solution_dir / 'sample_out.txt')
                for p in found.get('images', []):
                    try:
                        shutil.copy(p, solution_dir / p.name)
                    except Exception:
                        pass
            finally:
                try:
                    shutil.rmtree(tmpd)
                except Exception:
                    pass
        if solution_file is not None:
            sol_bytes = await solution_file.read(); sol_text = sol_bytes.decode('utf-8', errors='replace'); sol_text = extract_python_from_markdown(sol_text) or sol_text.strip(); save_solution_on_server(sol_text, solution_dir, {'solution_id': solution_id, 'provided_file': True})
        elif solution is not None:
            sol_text = extract_python_from_markdown(solution) or solution.strip(); save_solution_on_server(sol_text, solution_dir, {'solution_id': solution_id, 'provided': True})
    else:
        if solution_id is None: raise HTTPException(status_code=400, detail='Provide solution_id or upload solution_file/solution text')
        solution_dir = SOLUTIONS_DIR / solution_id
        if not solution_dir.exists(): raise HTTPException(status_code=404, detail='solution_id not found')

    if test_file is not None:
        test_input_text = (await test_file.read()).decode('utf-8')
    else:
        test_input_text = test_input

    coding_path = solution_dir / 'coding_solution.py'
    if not coding_path.exists():
        raise HTTPException(status_code=404, detail='coding_solution.py not found for this solution_id')
    current_code = coding_path.read_text(encoding='utf-8')

    run_res = run_python_code_str(current_code, test_input_text, timeout=EXECUTION_TIMEOUT)

    def normalize_out(s: str) -> str:
        return '\n'.join(line.rstrip() for line in s.strip().splitlines())

    if not run_res['timed_out'] and run_res['stderr'] == '' and (test_expected is None or (not check) or normalize_out(run_res['stdout']) == normalize_out(test_expected)):
        (solution_dir / 'test_output.txt').write_text(run_res['stdout'], encoding='utf-8')
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': run_res['stdout'], 'test_stderr': run_res['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'solution': current_code})

    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + '\n[stderr]\n' + run_res['stderr'], encoding='utf-8')

    if not GOOGLE_API_KEY:
        return JSONResponse({'status': 'failed', 'reason': 'no_google_api_key', 'run_result': run_res, 'solution': current_code}, status_code=400)

    try:
        import google.genai as genai
    except Exception as e:
        return JSONResponse({'status': 'failed', 'reason': f'missing_llm_lib: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)
    genai.configure(api_key=GOOGLE_API_KEY); model = genai.GenerativeModel(MODEL_NAME)

    last_code = current_code; last_error = f"Initial run failed. stdout:\n{run_res['stdout']}\nstderr:\n{run_res['stderr']}"

    images_in_dir = [p for p in solution_dir.iterdir() if p.is_file() and (('image' in p.name.lower()) or ('photo' in p.name.lower()) or p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.txt'])][:MAX_IMAGES_IN_PROMPT]
    image_urls = []; downloaded_local_paths = []
    for p in images_in_dir:
        if p.suffix.lower() in ['.txt', '.uri', '.url']:
            uri = read_image_uri_text(p)
            if uri.startswith('http://') or uri.startswith('https://'):
                image_urls.append(uri)
                if ENABLE_IMAGE_DOWNLOAD:
                    tmpd = Path(tempfile.mkdtemp()); local_path = tmpd / (p.stem + os.path.splitext(uri.split('?')[0])[1] if os.path.splitext(uri.split('?')[0])[1] else p.stem + '.jpg');
                    if download_image_from_url(uri, local_path): downloaded_local_paths.append(local_path)
        else:
            downloaded_local_paths.append(p)

    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else (statement or "")
    statement_text_replaced, photo_placeholders = extract_photo_placeholders(statement_text)
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else (sample_in or "")
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else (sample_out or "")

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt_text = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed on a later test input. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{statement_text_replaced}

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

If a corrected solution is provided, reply with the full Python code in a single markdown code block (python ... ).
"""

        # resp = None
        # if MULTIMODAL_MODE and (image_urls or downloaded_local_paths):
        #     try:
        #         resp = model.generate_content(prompt_text, referenced_image_urls=image_urls)
        #     except Exception:
        #         try:
        #             resp = model.generate_content(prompt_text, image_urls=image_urls)
        #         except Exception:
        #             pass
        #     if resp is None and downloaded_local_paths:
        #         try:
        #             image_bytes = [p.read_bytes() for p in downloaded_local_paths]
        #             try:
        #                 resp = model.generate_content(prompt_text, referenced_images=image_bytes)
        #             except Exception:
        #                 resp = model.generate_content(prompt_text, images=image_bytes)
        #         except Exception:
        #             pass
        if pil_images or image_urls or photo_placeholders:
            prompt_text += "\nI will also prov/ide you the images in same order as that of their occurance. Make sure gather sufficient insights from them and approach the problem."
  
        if resp is None:
            image_map_text = ''
            if photo_placeholders:
                map_lines = []
                for ph in photo_placeholders:
                    pid = ph.get('id'); match = None
                    for p in images_in_dir:
                        if pid in p.name: match = p.name; break
                    map_lines.append(f"PHOTO_ID {pid} maps to file {match or '[no-match]'}")
                image_map_text += '\n'.join(map_lines)
            if image_urls: image_map_text += '\nImage URLs:\n' + '\n'.join(image_urls)
            if image_map_text: prompt_text = prompt_text + '\n\n' + image_map_text
            try:
                resp = model.generate_content(prompt_text)
            except Exception as e:
                return JSONResponse({'status': 'failed', 'reason': f'LLM_call_failed: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)

        raw_llm_text = getattr(resp, 'text', str(resp)) or ""
        code_candidate = extract_python_from_markdown(raw_llm_text) or (raw_llm_text or "").strip()
        last_code = code_candidate

        if sample_in_text and sample_out_text:
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = '\n'.join(line.rstrip() for line in sample_run['stdout'].strip().splitlines())
            expected_norm = '\n'.join(line.rstrip() for line in sample_out_text.strip().splitlines())
            if sample_run['timed_out']:
                last_error = f"Sample run timed out: {sample_run['stderr']}"; continue
            if sample_run['stderr']:
                last_error = f"Sample runtime error after regen: {sample_run['stderr']}"; continue
            if check and sample_out_norm != expected_norm:
                diff = ''.join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch after regen. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"; continue

        test_run = run_python_code_str(code_candidate, test_input_text, timeout=EXECUTION_TIMEOUT)
        test_out_norm = '\n'.join(line.rstrip() for line in test_run['stdout'].strip().splitlines())
        if test_run['timed_out']:
            last_error = f"Test run timed out: {test_run['stderr']}"; continue
        if test_run['stderr']:
            last_error = f"Test runtime error after regen: {test_run['stderr']}"; continue
        if test_expected is not None:
            expected_test_norm = '\n'.join(line.rstrip() for line in test_expected.strip().splitlines())
            if test_out_norm != expected_test_norm:
                diff = ''.join(difflib.unified_diff(expected_test_norm.splitlines(keepends=True), test_out_norm.splitlines(keepends=True), fromfile='expected_test', tofile='actual_test'))
                last_error = f"Test mismatch after regen. Diff:\n{diff}\nStdout:\n{test_run['stdout']}\nStderr:\n{test_run['stderr']}"; continue
        else:
            if test_out_norm == "":
                last_error = "Test run produced empty stdout after regen."; continue

        coding_path.write_text(code_candidate, encoding='utf-8')
        (solution_dir / 'test_output.txt').write_text(test_run['stdout'], encoding='utf-8')
        (solution_dir / 'llm_response.txt').write_text(raw_llm_text, encoding='utf-8')
        (solution_dir / 'metadata.json').write_text(str({'solution_id': solution_id, 'regenerated_attempt': attempt}), encoding='utf-8')
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': test_run['stdout'], 'test_stderr': test_run['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'solution': code_candidate, 'attempts': attempt})

    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + '\n[stderr]\n' + run_res['stderr'], encoding='utf-8')
    return JSONResponse({'status': 'failed', 'reason': 'regeneration_exhausted', 'last_error': last_error, 'last_solution': last_code, 'solution': last_code}, status_code=400)


@app.get('/download/{solution_id}/{filename}')
def download_file(solution_id: str, filename: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail='solution_id not found')
    target = (solution_dir / filename).resolve()
    if not str(target).startswith(str(solution_dir.resolve()) + os.sep):
        raise HTTPException(status_code=400, detail='Invalid filename')
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(str(target), media_type='application/octet-stream', filename=target.name)


@app.get('/solutions/{solution_id}/files')
def list_solution_files(solution_id: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail='solution_id not found')
    files = [p.name for p in solution_dir.iterdir() if p.is_file()]
    return {'solution_id': solution_id, 'files': files}


@app.get('/solutions')
def list_solutions():
    ids = []
    for p in SOLUTIONS_DIR.iterdir():
        if p.is_dir():
            ids.append({'solution_id': p.name, 'files': [f.name for f in p.iterdir() if f.is_file()]})
    return {'solutions': ids}



# import requests
# url = "[https://my-codegen-api2.onrender.com/generate](https://my-codegen-api2.onrender.com/generate)"
# files = {"file": open("MyQ.zip", "rb")}
# r = requests.post(url, files=files, timeout=120)
# print(r.status_code)
# print(r.text)
# open("gen_response.json","wb").write(r.content)

# $Url = "[https://my-codegen-api2.onrender.com](https://my-codegen-api2.onrender.com)"
# curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
# Get-Content gen_response.json -Raw | Out-File -FilePath gen_response_pretty.json


# r = requests.post("[https://my-codegen-api2.onrender.com/test](https://my-codegen-api2.onrender.com/test)", data={"solution_id": "<SOLUTION_ID>"}, files={"test_file": open("test_input.txt","rb")})
# open("test_response.json","wb").write(r.content)
# print(r.text)
# curl.exe -s -X POST "$Url/test" -F "solution_id=<SOLUTION_ID>" -F "test_file=@test_input.txt" -o test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List


# r = requests.get(f"[https://my-codegen-api2.onrender.com/download/](https://my-codegen-api2.onrender.com/download/)<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)
# r = requests.get(f"[https://my-codegen-api2.onrender.com/download/](https://my-codegen-api2.onrender.com/download/)<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)











# import requests
# url = "https://my-codegen-api2.onrender.com/generate"
# files = {"file": open("MyQ.zip", "rb")}
# r = requests.post(url, files=files, timeout=120)
# print(r.status_code)
# print(r.text)
# open("gen_response.json","wb").write(r.content)

# $Url = "https://my-codegen-api2.onrender.com"
# curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
# Get-Content gen_response.json -Raw | Out-File -FilePath gen_response_pretty.json




# r = requests.post("https://my-codegen-api2.onrender.com/test", data={"solution_id": "<SOLUTION_ID>"}, files={"test_file": open("test_input.txt","rb")})
# open("test_response.json","wb").write(r.content)
# print(r.text)

# curl.exe -s -X POST "$Url/test" -F "solution_id=<SOLUTION_ID>" -F "test_file=@test_input.txt" -o test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List



# r = requests.get(f"https://my-codegen-api2.onrender.com/download/<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)

# curl.exe -s -X GET "$Url/download/<SOLUTION_ID>/coding_solution.py" -o coding_solution.py




#----------------------------------------------------------------------------------------------

@app.delete("/cleanup/{solution_id}")
def cleanup_solution(solution_id: str):
    """
    Safely remove all artifacts for a solution.
    Deletes the directory SOLUTIONS_DIR/<solution_id> and its contents.
    """
    solution_dir = (SOLUTIONS_DIR / solution_id).resolve()
    base = SOLUTIONS_DIR.resolve()

    # Ensure we are deleting only inside SOLUTIONS_DIR
    if not str(solution_dir).startswith(str(base) + os.sep) and solution_dir != base:
        raise HTTPException(status_code=400, detail="Invalid solution_id / path")

    if not solution_dir.exists():
        # 404: nothing to delete
        raise HTTPException(status_code=404, detail="solution_id not found")

    try:
        shutil.rmtree(solution_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove solution files: {e}")

    return JSONResponse({"status": "ok", "solution_id": solution_id, "message": "solution files removed"}, status_code=status.HTTP_200_OK)


# curl -X DELETE http://localhost:8000/cleanup/your_solution_id
# import requests

# solution_id = "your_solution_id"
# url = f"http://localhost:8000/cleanup/{solution_id}"

# response = requests.delete(url)

# print(response.status_code)
# print(response.json())

@app.post("/test2")
async def test2_endpoint_json(payload: Dict[str, Any]):
    """
    New /test2 endpoint to receive local-run results (client posts JSON).
    Accepts same fields as our local-run script posts:
      - solution_id (optional if 'solution' provided)
      - solution (optional code text to save)
      - stdout, stderr, timed_out (bool), returncode, test_input (string), test_expected (optional)
    Behavior:
      - Save run output to solution dir
      - If successful (no stderr and not timed_out and test_expected matches) -> status ok
      - Otherwise, behave like /test: call LLM (multimodal preferred) up to MAX_GENERATION_ATTEMPTS and return regenerated candidate
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
    if provided_solution_text and not solution_id:
        solution_id = str(uuid.uuid4())
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

    if (not timed_out) and stderr.strip() == "" and (test_expected is None or _normalize_out(stdout) == _normalize_out(test_expected)):
        return JSONResponse({"status": "ok", "solution_id": solution_id, "test_stdout": stdout, "test_stderr": stderr})

    last_code = coding_path.read_text(encoding="utf-8") if coding_path.exists() else (provided_solution_text or "")
    last_error = f"Local run failed. timed_out={timed_out}\nreturncode={returncode}\nstderr:\n{stderr}\nstdout:\n{stdout}"

    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else ""
    statement_text_replaced, photo_placeholders = (extract_photo_placeholders(statement_text) if statement_text else (statement_text, []))
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else ""
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else ""

    # discover images in solution_dir
    images_found = []
    for p in solution_dir.iterdir():
        if p.is_file():
            if ('image' in p.name.lower()) or ('photo' in p.name.lower()) or p.suffix.lower() in ['.png','.jpg','.jpeg','.gif','.bmp','.webp','.txt','.uri','.url','.link']:
                images_found.append(p)

    image_urls = []
    downloaded_local_paths = []
    pil_images = []
    multimodal_errors = []
    tmp_dirs = []

    for p in images_found:
        if p.suffix.lower() in ['.txt', '.uri', '.url', '.link']:
            uri = _read_image_uri_text_for_test2(p)
            if uri.startswith('http://') or uri.startswith('https://'):
                image_urls.append(uri)
                if _requests_if_available is not None:
                    tmpd = Path(tempfile.mkdtemp())
                    tmp_dirs.append(tmpd)
                    ext = ''
                    try:
                        ext = Path(uri.split('?')[0]).suffix or ''
                    except Exception:
                        ext = ''
                    local_name = (p.stem + ext) if ext else p.stem + '.jpg'
                    local_path = tmpd / local_name
                    ok = _download_image_from_url_for_test2(uri, local_path)
                    if ok:
                        downloaded_local_paths.append(local_path)
                    else:
                        multimodal_errors.append(f"download failed for {uri}")
                else:
                    multimodal_errors.append("requests not installed; cannot download image URLs")
            else:
                image_urls.append(uri)
        else:
            downloaded_local_paths.append(p)

    # try to open PIL images
    if PIL_Image_for_test2 is not None:
        for p in downloaded_local_paths:
            try:
                img = PIL_Image_for_test2.open(str(p)); img.load()
                pil_images.append(img)
            except Exception as e:
                multimodal_errors.append(f"PIL open failed {p.name}: {e}")
    else:
        if downloaded_local_paths:
            multimodal_errors.append("PIL not installed; cannot open local images")

    # configure LLM
    if not GOOGLE_API_KEY:
        (solution_dir / "metadata.json").write_text(json.dumps({"solution_id": solution_id, "last_error": last_error, "timestamp": time.time()}), encoding="utf-8")
        return JSONResponse({"status": "failed", "reason": "no_google_api_key", "last_error": last_error}, status_code=400)

    try:
        import google.genai as genai
    except Exception as e:
        return JSONResponse({"status": "failed", "reason": f"missing_llm_lib: {e}", "last_error": last_error}, status_code=500)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    raw_llm_text = None
    last_solution = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed when run locally. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{statement_text_replaced}

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
        resp = None
        attempt_errors = []
        if pil_images or image_urls or photo_placeholders:
            prompt += "\nI will also prov/ide you the images in same order as that of their occurance. Make sure gather sufficient insights from them and approach the problem."
  

        ## 1) try PIL prompt_parts
        # if MULTIMODAL_MODE and pil_images:
        #     try:
        #         prompt_parts = [prompt] + [im for im in pil_images[:MAX_IMAGES_IN_PROMPT]]
        #         resp = model.generate_content(prompt_parts)
        #         raw_llm_text = getattr(resp, "text", str(resp)) or ""
        #     except Exception as e:
        #         attempt_errors.append(f"prompt_parts failed: {repr(e)}")
        #         resp = None

        # # 2) try referenced_image_urls / image_urls
        # if resp is None and MULTIMODAL_MODE and image_urls:
        #     try:
        #         resp = model.generate_content(prompt, referenced_image_urls=image_urls)
        #         raw_llm_text = getattr(resp, "text", str(resp)) or ""
        #     except Exception:
        #         try:
        #             resp = model.generate_content(prompt, image_urls=image_urls)
        #             raw_llm_text = getattr(resp, "text", str(resp)) or ""
        #         except Exception as e:
        #             attempt_errors.append(f"image_urls calls failed: {repr(e)}")
        #             resp = None

        ## 3) try image bytes
        # if resp is None and downloaded_local_paths:
        #     try:
        #         image_bytes = [p.read_bytes() for p in downloaded_local_paths[:MAX_IMAGES_IN_PROMPT]]
        #         try:
        #             resp = model.generate_content(prompt, referenced_images=image_bytes)
        #             raw_llm_text = getattr(resp, "text", str(resp)) or ""
        #         except Exception:
        #             resp = model.generate_content(prompt, images=image_bytes)
        #             raw_llm_text = getattr(resp, "text", str(resp)) or ""
        #     except Exception as e:
        #         attempt_errors.append(f"bytes-based generate failed: {repr(e)}")
        #         resp = None

        # 4) text-only fallback with image mapping
        if resp is None:
            image_map_lines = []
            if photo_placeholders:
                for ph in photo_placeholders:
                    pid = ph.get('id'); match = None
                    for p in downloaded_local_paths + list(solution_dir.iterdir()):
                        if pid in p.name:
                            match = p.name; break
                    image_map_lines.append(f"PHOTO_ID {pid} maps to file {match or '[no-match]'}")
            if image_urls:
                image_map_lines.append("\nImage URLs:\n" + "\n".join(image_urls))
            mapping_text = ("\n-- Image mapping (attachment order) --\n" + "\n".join(image_map_lines)) if image_map_lines else ""
            prompt_with_map = prompt + ("\n\n" + mapping_text if mapping_text else "")
            try:
                resp = model.generate_content(prompt_with_map)
                raw_llm_text = getattr(resp, "text", str(resp)) or ""
            except Exception as e:
                attempt_errors.append(f"text-only generate failed: {repr(e)}")
                resp = None

        if resp is None:
            last_error = f"LLM generate all methods failed on attempt {attempt}; errors: {attempt_errors + multimodal_errors}"
            continue

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

        # success - save candidate and return
        try:
            coding_path.write_text(code_candidate, encoding='utf-8')
            (solution_dir / 'test_output.txt').write_text(stdout + "\n[stderr]\n" + stderr, encoding='utf-8')
            (solution_dir / 'llm_response.txt').write_text(raw_llm_text, encoding='utf-8')
            (solution_dir / 'metadata.json').write_text(json.dumps({"solution_id": solution_id, "regenerated_attempt": attempt, "timestamp": time.time()}), encoding='utf-8')
        except Exception:
            pass

        for d in tmp_dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass

        return JSONResponse({"status": "regenerated", "solution_id": solution_id, "candidate_solution": code_candidate, "raw_llm_text": raw_llm_text, "attempts": attempt})

    # exhausted
    for d in tmp_dirs:
        try:
            shutil.rmtree(d)
        except Exception:
            pass

    return JSONResponse({"status": "failed", "reason": "regeneration_exhausted", "last_error": last_error, "last_solution": last_solution, "raw_llm_text": raw_llm_text}, status_code=400)
# ---- end /test2 endpoint



























#----------------------------------------------------------------------------------------------


# reuse existing names from your server file: SOLUTIONS_DIR, EXECUTION_TIMEOUT, MAX_GENERATION_ATTEMPTS, MODEL_NAME, GOOGLE_API_KEY
# reuse helpers: save_solution_on_server, extract_python_from_markdown, find_problem_files


# # --------------------
# # Local runner: save this as test2_runner.py in the same directory as the generated program (or adapt filenames)
# # The server returns an auto-generated version of this script via /test2. You can also copy/save the script below.
# # --------------------

# # test2_runner.py
# TEST2_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
# import sys
# import subprocess
# import json
# import requests
# import traceback
# from pathlib import Path

# # Edit these values before running
# SERVER_BASE = ''  # e.g. 'https://myserver.example.com'
# REPORT_PATH = '/test2/result'
# SOLUTION_ID = ''
# PROGRAM_FILENAME = 'a.py'  # change if your local program filename differs
# INPUT_FILENAME = 'test_input.txt'
# TIME_LIMIT = 60

# # read input from INPUT_FILENAME if present, else empty
# if Path(INPUT_FILENAME).exists():
#     test_input = Path(INPUT_FILENAME).read_text(encoding='utf-8')
# else:
#     test_input = ''

# stdout = ''
# stderr = ''
# timed_out = False
# returncode = None
# try:
#     with open(INPUT_FILENAME, 'w', encoding='utf-8') as f:
#         f.write(test_input)
#     with open(INPUT_FILENAME, 'r', encoding='utf-8') as inf:
#         p = subprocess.run([sys.executable, PROGRAM_FILENAME], stdin=inf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=TIME_LIMIT)
#         stdout = p.stdout
#         stderr = p.stderr
#         returncode = p.returncode
# except subprocess.TimeoutExpired:
#     timed_out = True
#     stderr = f'Time Limit Exceeded ({TIME_LIMIT} seconds)'
# except Exception as e:
#     stderr = f'Local runner exception: {repr(e)}\n' + traceback.format_exc()

# payload = {
#     'solution_id': SOLUTION_ID,
#     'stdout': stdout,
#     'stderr': stderr,
#     'timed_out': timed_out,
#     'returncode': returncode,
#     'test_input': test_input
# }

# if not SERVER_BASE:
#     print('Edit SERVER_BASE and SOLUTION_ID and run again to POST results to server')
#     print('Payload preview:')
#     print(json.dumps(payload)[:2000])
#     sys.exit(0)

# url = SERVER_BASE.rstrip('/') + REPORT_PATH
# try:
#     r = requests.post(url, json=payload, timeout=60)
#     print('Server response code:', r.status_code)
#     try:
#         print('Response JSON:', r.json())
#     except Exception:
#         print('Response text:', r.text[:2000])
# except Exception as e:
#     print('Failed to POST results to server:', e)
#     print('Payload saved to latest_run_payload.json')
#     open('latest_run_payload.json','w', encoding='utf-8').write(json.dumps(payload))
# '''

# # End of generated code file




















# import requests
# url = "[https://my-codegen-api2.onrender.com/generate](https://my-codegen-api2.onrender.com/generate)"
# files = {"file": open("MyQ.zip", "rb")}
# r = requests.post(url, files=files, timeout=120)
# print(r.status_code)
# print(r.text)
# open("gen_response.json","wb").write(r.content)

# $Url = "[https://my-codegen-api2.onrender.com](https://my-codegen-api2.onrender.com)"
# curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
# Get-Content gen_response.json -Raw | Out-File -FilePath gen_response_pretty.json


# r = requests.post("[https://my-codegen-api2.onrender.com/test](https://my-codegen-api2.onrender.com/test)", data={"solution_id": "<SOLUTION_ID>"}, files={"test_file": open("test_input.txt","rb")})
# open("test_response.json","wb").write(r.content)
# print(r.text)
# curl.exe -s -X POST "$Url/test" -F "solution_id=<SOLUTION_ID>" -F "test_file=@test_input.txt" -o test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List


# r = requests.get(f"[https://my-codegen-api2.onrender.com/download/](https://my-codegen-api2.onrender.com/download/)<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)
# r = requests.get(f"[https://my-codegen-api2.onrender.com/download/](https://my-codegen-api2.onrender.com/download/)<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)











# import requests
# url = "https://my-codegen-api2.onrender.com/generate"
# files = {"file": open("MyQ.zip", "rb")}
# r = requests.post(url, files=files, timeout=120)
# print(r.status_code)
# print(r.text)
# open("gen_response.json","wb").write(r.content)

# $Url = "https://my-codegen-api2.onrender.com"
# curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
# Get-Content gen_response.json -Raw | Out-File -FilePath gen_response_pretty.json




# r = requests.post("https://my-codegen-api2.onrender.com/test", data={"solution_id": "<SOLUTION_ID>"}, files={"test_file": open("test_input.txt","rb")})
# open("test_response.json","wb").write(r.content)
# print(r.text)

# curl.exe -s -X POST "$Url/test" -F "solution_id=<SOLUTION_ID>" -F "test_file=@test_input.txt" -o test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List



# r = requests.get(f"https://my-codegen-api2.onrender.com/download/<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)

# curl.exe -s -X GET "$Url/download/<SOLUTION_ID>/coding_solution.py" -o coding_solution.py













#HELPER
# #!/usr/bin/env python3
# """
# Higher-level automate script that prepares (optionally uploads) and runs locally,
# and only contacts server on error.

# Usage examples:

# # Use an existing solution_id on server, provide local test file:
# python test2_automate_post_on_error.py --server https://my-api-mneh.onrender.com --solution-id <ID> --test-file pp_input.txt

# # Upload local code + test file in one call (server will store it if you post failures):
# python test2_automate_post_on_error.py --server https://my-api-mneh.onrender.com --solution-file a.py --test-file pp_input.txt

# """
# import argparse
# import requests
# import time
# import os
# import sys
# import json
# import shutil
# import subprocess
# from pathlib import Path

# DOWNLOAD_TIMEOUT = 60
# DEFAULT_MAX_ITERS = 4
# DEFAULT_TIMEOUT = 60

# def call_prepare(server, solution_id=None, solution_file=None, test_file=None, test_input_str=None):
#     """
#     Prepare step: if solution_file present, we don't yet POST to server.
#     We return a dict which may include 'solution_id' and 'program_filename' if server had one.
#     For this simplified flow we just return a minimal dict if we didn't call server.
#     """
#     # If we have solution_id and want to retrieve program metadata, attempt to call /solutions/{id}/files
#     if solution_id:
#         try:
#             r = requests.get(server.rstrip('/') + f'/solutions/{solution_id}/files', timeout=DOWNLOAD_TIMEOUT)
#             r.raise_for_status()
#             return r.json()
#         except Exception:
#             # ignore; calling server not mandatory here
#             return {}
#     return {}

# def download_program(server, solution_id, program_filename, out_path):
#     url = server.rstrip('/') + f'/download/{solution_id}/{program_filename}'
#     r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
#     r.raise_for_status()
#     with open(out_path, 'wb') as f:
#         shutil.copyfileobj(r.raw, f)
#     return out_path

# def run_program_locally(python_exe, program_path, test_input_path, time_limit):
#     try:
#         with open(test_input_path, 'r', encoding='utf-8') as inf:
#             p = subprocess.run([python_exe, str(program_path)], stdin=inf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=time_limit)
#             return {"stdout": p.stdout, "stderr": p.stderr, "timed_out": False, "returncode": p.returncode}
#     except subprocess.TimeoutExpired:
#         return {"stdout": "", "stderr": f"Time Limit Exceeded ({time_limit} seconds)", "timed_out": True, "returncode": None}
#     except Exception as e:
#         return {"stdout": "", "stderr": f"Local runner exception: {repr(e)}", "timed_out": False, "returncode": None}

# def post_result(server, payload):
#     url = server.rstrip('/') + '/test2'
#     r = requests.post(url, json=payload, timeout=DOWNLOAD_TIMEOUT)
#     r.raise_for_status()
#     return r.json()

# def normalize_out(s: str) -> str:
#     return "\n".join(line.rstrip() for line in s.strip().splitlines())

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--server', required=True)
#     p.add_argument('--solution-id', help='Existing solution_id on the server (optional)')
#     p.add_argument('--solution-file', help='Local solution code file to upload (optional)')
#     p.add_argument('--test-file', help='Local test input file (required unless --test-input is passed)')
#     p.add_argument('--test-input', help='Direct test input string (alternative)')
#     p.add_argument('--program-filename', default='coding_solution.py', help='Name to save downloaded program locally')
#     p.add_argument('--python-exe', default=sys.executable, help='Python executable to run locally')
#     p.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS)
#     p.add_argument('--time-limit', type=int, default=DEFAULT_TIMEOUT)
#     p.add_argument('--upload-local-program', action='store_true', help="Include program code in POST when failing")
#     args = p.parse_args()

#     if not args.test_file and args.test_input is None:
#         print("Either --test-file or --test-input must be provided.", file=sys.stderr)
#         sys.exit(2)

#     # If solution_file provided, use it locally; else attempt to download if solution_id provided
#     if args.solution_file:
#         local_program_path = Path(args.solution_file)
#         if not local_program_path.exists():
#             print("solution_file not found:", local_program_path); sys.exit(2)
#     elif args.solution_id:
#         # attempt download
#         try:
#             print("Attempting to download program from server...")
#             download_program(args.server, args.solution_id, args.program_filename, Path(args.program_filename))
#             local_program_path = Path(args.program_filename)
#             print("Downloaded program to", local_program_path)
#         except Exception as e:
#             print("Failed to download program:", e)
#             print("Provide --solution-file if you have a local copy.")
#             sys.exit(1)
#     else:
#         print("Provide either --solution-file or --solution-id to obtain a program to run.", file=sys.stderr)
#         sys.exit(2)

#     # ensure test input exists
#     if args.test_file:
#         local_test_input = Path(args.test_file)
#         if not local_test_input.exists():
#             print("test file not found:", local_test_input); sys.exit(2)
#     else:
#         local_test_input = Path('test_input_temp.txt')
#         local_test_input.write_text(args.test_input or '', encoding='utf-8')

#     # Run locally first; only call server if failure
#     for attempt in range(1, args.max_iters + 1):
#         print(f"\n=== ITER {attempt}/{args.max_iters} ===")
#         run_res = run_program_locally(args.python_exe, local_program_path, local_test_input, args.time_limit)
#         print("Local run: timed_out=", run_res['timed_out'], " returncode=", run_res['returncode'])
#         Path('latest_run_stdout.txt').write_text(run_res['stdout'] or '', encoding='utf-8')
#         Path('latest_run_stderr.txt').write_text(run_res['stderr'] or '', encoding='utf-8')

#         success = (not run_res['timed_out']) and (not run_res['stderr'].strip()) and (run_res['returncode'] in (0, None))
#         if success:
#             print("Success locally. No server contact required. Exiting.")
#             sys.exit(0)

#         # Local failure -> post to server /test2
#         payload = {
#             'solution_id': args.solution_id,
#             'stdout': run_res['stdout'] or '',
#             'stderr': run_res['stderr'] or '',
#             'timed_out': bool(run_res['timed_out']),
#             'returncode': run_res['returncode'],
#             'test_input': local_test_input.read_text(encoding='utf-8')
#         }
#         if args.upload_local_program:
#             payload['solution'] = local_program_path.read_text(encoding='utf-8')

#         print("Posting failure to server /test2 ...")
#         try:
#             server_resp = post_result(args.server, payload)
#         except Exception as e:
#             print("Failed to post to server:", e); sys.exit(1)

#         status = server_resp.get('status')
#         if status == 'regenerated':
#             candidate = server_resp.get('candidate_solution') or server_resp.get('solution') or server_resp.get('candidate')
#             if not candidate:
#                 candidate = server_resp.get('raw_llm_text') or ""
#             if not candidate:
#                 print("No candidate returned by server. Response:", server_resp); sys.exit(1)
#             print("Saving regenerated candidate to", local_program_path)
#             local_program_path.write_text(candidate, encoding='utf-8')
#             if 'solution_id' in server_resp:
#                 args.solution_id = server_resp['solution_id']
#             continue
#         elif status == 'ok':
#             print("Server returned ok (candidate verified server-side). Exiting.")
#             sys.exit(0)
#         else:
#             print("Server returned failure. Response:", server_resp)
#             sys.exit(1)

#     print("Reached max iterations without success.")
#     sys.exit(2)

# if __name__ == "__main__":
#     main()
