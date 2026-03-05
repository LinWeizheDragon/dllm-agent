from fastapi import FastAPI, HTTPException
import types
from pydantic import BaseModel as PydanticBaseModel
from typing import List, Optional, Dict, Union
import torch
from datetime import datetime
import uuid
import argparse

from inference.generation_utils import diffusion_generate
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="LLM API", description="OpenAI compatible API")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load", type=str, required=True)
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--device", type=str, default="npu")
    return p.parse_args()

ARGS = parse_args()

MODEL_PATH = ARGS.load
device = ARGS.device

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model.diffusion_generate = types.MethodType(diffusion_generate, model)

mask_token_id = 45830
eos_token_id = tokenizer.eos_token_id


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


@app.get("/")
def read_root():
    return {"message": "LLM API running"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        prompt_text = ""
        for msg in request.messages:
            if msg.role == "user":
                prompt_text = msg.content
                break

        if not prompt_text:
            raise HTTPException(status_code=400, detail="missing user message")

        gen_length = request.max_tokens or 128
        if gen_length % 32 != 0:
            raise HTTPException(status_code=400, detail="max_tokens must be multiple of 32")

        chat_template = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        user_input = tokenizer.apply_chat_template(
            chat_template,
            add_generation_prompt=True,
            tokenize=False
        )

        input_ids = tokenizer(user_input)["input_ids"]
        prompt = torch.tensor(input_ids).to(device).unsqueeze(0)
        attention_mask = prompt.ne(tokenizer.pad_token_id)

        temperature = request.temperature or 0.0
        top_p = request.top_p or 1.0

        out = model.diffusion_generate(
            prompt,
            block_length=32,
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=gen_length,
            alg="confidence_threshold",
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            num_small_blocks=4
        )

        response_text = tokenizer.batch_decode(
            out[:, prompt.shape[1]:]
        )[0]

        response_text = response_text.split(tokenizer.eos_token)[0].strip()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                    "token_ids": None
                }
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "local-model",
                "object": "model",
                "created": 1677652288,
                "owned_by": "local"
            }
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "local-model",
        "openai_compatible": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
