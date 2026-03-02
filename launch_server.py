# main.py
"""
A lightweight FastAPI service for loading a Diffusion LLM and serving
OpenAI-compatible Chat Completions (non-streaming) for local inference.

This script:
- Loads a HuggingFace-style model/tokenizer from a local path
- Attaches a custom `diffusion_generate` method to the model
- Exposes minimal endpoints: /v1/chat/completions, /v1/models, /health
"""

from __future__ import annotations

import argparse
import types
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel
import uvicorn

from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.generation_utils import diffusion_generate


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Local Diffusion LLM API Server",
    description="A lightweight OpenAI-compatible Chat Completions service for local Diffusion LLM inference.",
)

# -----------------------------------------------------------------------------
# Global runtime objects (loaded once at startup)
# -----------------------------------------------------------------------------
ARGS = None
MODEL = None
TOKENIZER = None
DEVICE = None
MASK_TOKEN_ID = None

# Public-facing model id returned by /v1/models and used in responses.
MODEL_ID = "local-diffusion-llm"


# -----------------------------------------------------------------------------
# OpenAI-compatible request/response schemas
# -----------------------------------------------------------------------------
class ChatMessage(PydanticBaseModel):
    role: str
    content: str


class ChatCompletionRequest(PydanticBaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False

    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1


class ChatCompletionResponseMessage(PydanticBaseModel):
    role: str
    content: str
    refusal: Optional[str] = None


class ChatCompletionResponseChoice(PydanticBaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str
    logprobs: Optional[Dict] = None
    token_ids: Optional[List[int]] = None


class ChatCompletionResponse(PydanticBaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _parse_torch_dtype(dtype_str: str):
    """Map user dtype string to torch dtype."""
    s = (dtype_str or "bfloat16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def load_model(model_path: str, device: str, torch_dtype: str = "bfloat16"):
    """
    Load model + tokenizer and attach the custom diffusion_generate method.

    Args:
        model_path: Local HF from_pretrained directory.
        device: Device string provided by the user (e.g., "npu:0", "cuda:0", "cpu").
        torch_dtype: One of {"bfloat16","float16","float32"}.
    """
    global MODEL, TOKENIZER, DEVICE, MASK_TOKEN_ID

    DEVICE = device
    dtype = _parse_torch_dtype(torch_dtype)

    # Use float32 on CPU to avoid dtype pitfalls.
    effective_dtype = dtype if DEVICE != "cpu" else torch.float32

    print(f"[INFO] Loading model from: {model_path}")
    print(f"[INFO] device={DEVICE}, dtype={effective_dtype}")

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=effective_dtype,
    ).to(DEVICE).eval()

    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Ensure pad_token_id exists to make attention_mask robust.
    if TOKENIZER.pad_token_id is None:
        if TOKENIZER.eos_token_id is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        else:
            TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})
            MODEL.resize_token_embeddings(len(TOKENIZER))

    # Bind custom generation function.
    MODEL.diffusion_generate = types.MethodType(diffusion_generate, MODEL)

    # Mask token id: prefer CLI flag.
    MASK_TOKEN_ID = int(ARGS.mask_token_id)

    print("[INFO] Model loaded successfully.")


@app.on_event("startup")
def _startup():
    # Load model once when the service starts.
    load_model(ARGS.load, device=ARGS.device, torch_dtype=ARGS.dtype)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "OpenAI-compatible local Diffusion LLM server is running."}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1677652288,
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
def health_check():
    loaded = MODEL is not None and TOKENIZER is not None
    return {
        "status": "OK" if loaded else "LOADING",
        "device": str(DEVICE),
        "model_path": getattr(ARGS, "load", None),
        "model_id": MODEL_ID,
        "openai_compatible": True,
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions endpoint (non-streaming).

    This server ignores request.model by default and always serves MODEL_ID.
    You can enforce strict equality if desired.
    """
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported in this implementation.")

        # Validate max_tokens. This implementation requires it to be a multiple of block_length.
        gen_length = int(request.max_tokens or 128)
        if gen_length <= 0:
            raise HTTPException(status_code=400, detail="max_tokens must be > 0")

        if gen_length % int(ARGS.block_length) != 0:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be a multiple of block_length={ARGS.block_length}, got {gen_length}.",
            )

        # Build text prompt using the tokenizer's chat template (if available).
        chat_template = [{"role": m.role, "content": m.content} for m in request.messages]
        user_input = TOKENIZER.apply_chat_template(
            chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize prompt.
        input_ids = TOKENIZER(user_input, add_special_tokens=False)["input_ids"]
        prompt = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
        attention_mask = prompt.ne(TOKENIZER.pad_token_id)

        temperature = float(request.temperature or 0.0)
        top_p = float(request.top_p or 1.0)

        eos_token_id = TOKENIZER.eos_token_id

        # Run custom diffusion generation.
        out = MODEL.diffusion_generate(
            prompt,
            block_length=int(ARGS.block_length),
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=gen_length,
            alg=str(ARGS.alg),
            mask_token_id=int(MASK_TOKEN_ID),
            eos_token_id=eos_token_id,
            num_small_blocks=int(ARGS.num_small_blocks),
        )

        # Decode only the newly generated part.
        response_text = TOKENIZER.batch_decode(out[:, prompt.shape[1] :], skip_special_tokens=False)[0]
        if TOKENIZER.eos_token is not None:
            response_text = response_text.split(TOKENIZER.eos_token)[0]
        response_text = response_text.strip()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=MODEL_ID,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(role="assistant", content=response_text, refusal=None),
                    finish_reason="stop",
                    logprobs=None,
                    token_ids=None,
                )
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        # Avoid leaking stack traces in a public service.
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenAI-compatible FastAPI server for local Diffusion LLM inference.")
    p.add_argument("--load", type=str, required=True, help="Path to model weights (HuggingFace from_pretrained dir).")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    p.add_argument("--port", type=int, default=9055, help="Port to listen on.")
    p.add_argument(
        "--device",
        type=str,
        default="npu:0",
        help='Device string, e.g. "npu:0", "cuda:0", or "cpu". Default: "npu:0".',
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    p.add_argument("--mask_token_id", type=int, default=45830, help="Mask token id for diffusion generation.")
    p.add_argument("--block_length", type=int, default=32, help="Block length for diffusion generation.")
    p.add_argument("--num_small_blocks", type=int, default=4, help="num_small_blocks for diffusion generation.")
    p.add_argument("--alg", type=str, default="confidence_threshold", help="Algorithm name for diffusion generation.")
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    print(f"[INFO] Starting server: http://{ARGS.host}:{ARGS.port}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)