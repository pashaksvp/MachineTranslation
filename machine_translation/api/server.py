from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from akkadian_mt.logging_utils import get_logger
from akkadian_mt.translator import MyTranslatorModel


logger = get_logger("akkadian_mt.api")
app = FastAPI(title="Akkadian Streaming Translator")
translator = MyTranslatorModel(
    model_name=os.getenv("MODEL_NAME", "google/byt5-small"),
    model_dir=os.getenv("MODEL_DIR", "model/baseline_raw"),
    normalize=os.getenv("NORMALIZE", "0") == "1",
    num_beams=int(os.getenv("NUM_BEAMS", "1")),
    max_source_length=int(os.getenv("MAX_SOURCE_LENGTH", "256")),
    max_target_length=int(os.getenv("MAX_TARGET_LENGTH", "128")),
)
tgi_url = os.getenv("TGI_URL", "").rstrip("/")
max_target_length = int(os.getenv("MAX_TARGET_LENGTH", "128"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")
log_path = Path("data/log_file.log")


class TranslateRequest(BaseModel):
    text: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/logs", response_class=PlainTextResponse)
def view_logs() -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


@app.get("/logs/download")
def download_logs() -> FileResponse:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    return FileResponse(log_path, filename="log_file.log", media_type="text/plain")


def _sse_event(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _tgi_stream(text: str) -> Iterator[str]:
    prompt = f"translate Akkadian to English: {text}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_target_length,
            "return_full_text": False,
        },
    }
    request = Request(
        f"{tgi_url}/generate_stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            token = payload.get("token", {})
            token_text = token.get("text", "")
            if token_text and not token.get("special", False):
                yield token_text


@app.post("/translate")
def translate(request: TranslateRequest) -> StreamingResponse:
    logger.info("Translation request received: %d chars", len(request.text))

    def stream() -> Iterator[str]:
        try:
            if tgi_url:
                for token in _tgi_stream(request.text):
                    yield _sse_event({"token": token})
            else:
                result = translator.predict(request.text, stream=True)
                if isinstance(result, str):
                    yield _sse_event({"token": result})
                else:
                    for token in result:
                        yield _sse_event({"token": token})
            yield _sse_event({"done": "true"})
        except Exception as exc:
            logger.exception("Translation failed")
            yield _sse_event({"error": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")
