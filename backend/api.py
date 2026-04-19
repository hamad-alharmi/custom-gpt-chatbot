import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from model import GPT, GPTConfig

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoint.pt"
enc        = tiktoken.get_encoding("gpt2")

def load_model():
    ckpt   = torch.load(CHECKPOINT, map_location=DEVICE)
    config = GPTConfig()
    for k, v in ckpt["config"].items():
        setattr(config, k, v)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded — val_loss {ckpt['val_loss']:.4f} | {model.param_count()/1e6:.1f}M params")
    return model

model = load_model()

app = FastAPI(title="Custom GPT API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message:        str
    history:        Optional[list] = []
    max_new_tokens: Optional[int]   = 200
    temperature:    Optional[float] = 0.8

class GenerateRequest(BaseModel):
    prompt:         str
    max_new_tokens: Optional[int]   = 200
    temperature:    Optional[float] = 0.8
    top_k:          Optional[int]   = 40

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.get("/model/info")
def info():
    return {
        "params_M":   round(model.param_count() / 1e6, 2),
        "n_layer":    model.config.n_layer,
        "n_head":     model.config.n_head,
        "n_embd":     model.config.n_embd,
        "block_size": model.config.block_size,
        "device":     DEVICE,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    prompt = ""
    for turn in req.history:
        prompt += f"{turn.get('role','user').upper()}: {turn.get('content','').strip()}\n"
    prompt += f"USER: {req.message}\nASSISTANT:"
    tokens  = enc.encode(prompt)
    max_ctx = model.config.block_size - req.max_new_tokens
    if len(tokens) > max_ctx:
        tokens = tokens[-max_ctx:]
    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    response = enc.decode(out[0][len(tokens):].tolist())
    if "USER:" in response:
        response = response[:response.index("USER:")]
    return {"role": "assistant", "content": response.strip()}

@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "prompt cannot be empty")
    tokens = enc.encode(req.prompt)
    idx    = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_k=req.top_k)
    return {
        "prompt":    req.prompt,
        "response":  enc.decode(out[0][len(tokens):].tolist()).strip(),
        "full_text": enc.decode(out[0].tolist()).strip(),
    }
