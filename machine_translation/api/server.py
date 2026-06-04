from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")


class TranslateRequest(BaseModel):
    text: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _sse_event(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/translate")
def translate(request: TranslateRequest) -> StreamingResponse:
    logger.info("Translation request received: %d chars", len(request.text))

    def stream() -> Iterator[str]:
        try:
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
