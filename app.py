# app.py
import os
import uuid
import shutil
import asyncio
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
TRANSCRIBER_SCRIPT = APP_DIR / "transcription_gpu.py"  # your script
HF_TOKEN = os.environ.get("HF_TOKEN")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GPU Transcription API (single-job)")

# simple CORS for demo; tune for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global single-job state
job_lock = asyncio.Lock()
job_running: bool = False
last_transcription_text: Optional[str] = None
last_transcription_format: Optional[str] = None
last_job_id: Optional[str] = None

# file names for persistent last transcription
LAST_TXT = OUTPUT_DIR / "last_transcription.txt"
LAST_JSON = OUTPUT_DIR / "last_transcription.json"

# if persistent last exists on start, load it
if LAST_TXT.exists():
    last_transcription_text = LAST_TXT.read_text(encoding="utf-8")
    last_transcription_format = "txt"
elif LAST_JSON.exists():
    last_transcription_text = LAST_JSON.read_text(encoding="utf-8")
    last_transcription_format = "json"


async def run_transcription_subprocess(input_path: Path, out_path: Path, model: str = "medium", out_format: str = "txt") -> int:
    """
    Calls the transcription_gpu.py script as a subprocess.
    Waits for it to finish. Returns exit code.
    """
    env = os.environ.copy()
    # ensure HF_TOKEN present
    if "HF_TOKEN" not in env or not env["HF_TOKEN"]:
        raise RuntimeError("HF_TOKEN must be set in environment")

    cmd = [
        "python3",
        str(TRANSCRIBER_SCRIPT),
        str(input_path),
        "--model", model,
        "-f", out_format,
        "-o",
    ]
    # transcription_gpu.py expects overwrite flag to avoid interactive; we'll pass it
    cmd.append("--overwrite")

    # ensure output filename is respected: transcription script writes file in cwd named after input
    # to be safe, run subprocess with cwd = OUTPUT_DIR and pass the input's full path in temp dir
    # We'll copy input into OUTPUT_DIR and run script there so output file is next to it
    tmp_input = OUTPUT_DIR / input_path.name
    shutil.copy2(input_path, tmp_input)

    proc = await asyncio.create_subprocess_exec(
        *["python3", str(TRANSCRIBER_SCRIPT), str(tmp_input), "--model", model, "--out_format", out_format, "--overwrite"],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(OUTPUT_DIR),
        env=env,
    )
    stdout, stderr = await proc.communicate()
    if stdout:
        print(stdout.decode(errors="ignore"))
    if stderr:
        # it's often noisy; keep it for logs
        print(stderr.decode(errors="ignore"))
    return proc.returncode


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), model: str = "medium", out_format: str = "txt"):
    """
    Accept an audio file and begin transcription if idle.
    Immediately returns 200 accepted if started, or 409 if busy.
    """
    global job_running, last_transcription_text, last_transcription_format, last_job_id

    # quick validation
    if out_format not in ("txt", "json"):
        raise HTTPException(status_code=400, detail="out_format must be 'txt' or 'json'")

    # try to acquire lock without waiting
    acquired = await job_lock.acquire() if not job_lock.locked() else False
    if not acquired:
        # busy
        return JSONResponse(status_code=409, content={"status": "busy", "message": "A transcription is already running."})

    # we now own the lock; spawn background task but return immediately 200
    job_running = True
    job_id = str(uuid.uuid4())
    last_job_id = job_id

    # save uploaded file
    # use original filename + uuid to avoid collisions
    filename = f"{job_id}_{Path(file.filename).name}"
    saved_path = UPLOADS_DIR / filename
    with saved_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    async def bg_task():
        global job_running, last_transcription_text, last_transcription_format
        try:
            # call the subprocess that runs transcription (blocks GPU work in other process)
            rc = await run_transcription_subprocess(saved_path, OUTPUT_DIR, model=model, out_format=out_format)
            if rc != 0:
                print(f"transcriber exited with code {rc}")
                # leave last_transcription as-is; you might want to store an error file
            else:
                # determine output filename created by transcription script:
                base = Path(saved_path).stem  # jobid_filename (maybe .wav was created)
                expected_ext = ".json" if out_format == "json" else ".txt"
                expected_out = OUTPUT_DIR / f"{base}{expected_ext}"
                if expected_out.exists():
                    text = expected_out.read_text(encoding="utf-8")
                    last_transcription_text = text
                    last_transcription_format = out_format
                    # persist to LAST_TXT/LAST_JSON for future runs
                    if out_format == "txt":
                        LAST_TXT.write_text(text, encoding="utf-8")
                    else:
                        LAST_JSON.write_text(text, encoding="utf-8")
                else:
                    print(f"expected output {expected_out} not found after successful exit")
        except Exception as e:
            print("Exception in bg_task:", e)
        finally:
            job_running = False
            # release the global lock so subsequent uploads can run
            try:
                job_lock.release()
            except RuntimeError:
                pass

    # schedule background task (fire-and-forget)
    asyncio.create_task(bg_task())

    return {"status": "accepted", "job": "running", "job_id": job_id}


@app.get("/status")
async def status():
    return {"status": "running" if job_running else "idle", "current_job_id": last_job_id}


#@app.get("/transcription")
#async def get_transcription():
#    if last_transcription_text is None:
#        return {"status": "no transcriptions"}
#    return {"status": "ready", "format": last_transcription_format, "transcription": last_transcription_text}


@app.get("/transcription")
async def get_transcription(format: Optional[str] = None):
    if last_transcription_text is None:
        return {"status": "no transcriptions"}

    if last_transcription_format == "txt":
        if format == "raw":
            # raw return
            return PlainTextResponse(last_transcription_text)

        # JSON wrapper but raw text preserved in a separate key
        return {
            "status": "ready",
            "format": "txt",
            "raw_text": last_transcription_text
        }

    return {
        "status": "ready",
        "format": "json",
        "transcription": last_transcription_text
    }

