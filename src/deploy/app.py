import logging
import os
from datetime import datetime
from uuid import uuid4
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

from src.deploy.config import (
    MODEL_NAME, BASE_URL, API_KEY,
    TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS, MODEL_TYPE
)
from src.deploy.model_loader import serve_model
LOG_FILE = "src/logs/llm.logs"

def log_event(message: str):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    full_msg = f"{timestamp} | {message}"

    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

    logger.info(message)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quantised LLM Server")

serve_model(MODEL_NAME, MODEL_TYPE)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

CHAT_SESSIONS: Dict[str, List[dict]] = {}

class LLMRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    chat_id: Optional[str] = None
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    max_tokens: int = MAX_TOKENS

def stream_llm(messages: List[dict], req: LLMRequest, request_id: str):
    log_event(f"[{request_id}] streaming | model={MODEL_NAME} | msgs={len(messages)}")

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        extra_body={"top_k": req.top_k},
        stream=True,
    )

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

def build_messages(system_prompt: Optional[str], user_prompt: str) -> List[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

@app.post("/generate")
def generate(req: LLMRequest):
    request_id = str(uuid4())
    messages = build_messages(req.system_prompt, req.prompt)

    log_event(f"[{request_id}] /generate started")

    return StreamingResponse(
        stream_llm(messages, req, request_id),
        media_type="text/plain",
    )

@app.post("/chat")
def chat(req: LLMRequest):
    request_id = str(uuid4())
    chat_id = req.chat_id or str(uuid4())
    history = CHAT_SESSIONS.setdefault(chat_id, [])
    if not history and req.system_prompt:
        history.append({"role": "system", "content": req.system_prompt})

    messages = history + [{"role": "user", "content": req.prompt}]
    log_event(f"[{request_id}] /chat started | chat_id={chat_id}")
    def generator():
        buffer = []

        for token in stream_llm(messages, req, request_id):
            buffer.append(token)
            yield token

        history.extend([
            {"role": "user", "content": req.prompt},
            {"role": "assistant", "content": "".join(buffer)},
        ])
        log_event(f"[{request_id}] /chat completed | chat_id={chat_id} | turns={len(history)}")

    return StreamingResponse(generator(), media_type="text/plain")
