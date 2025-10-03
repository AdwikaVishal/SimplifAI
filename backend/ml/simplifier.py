# simplifier.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
from fastapi import HTTPException
import os

app = FastAPI()

# Allow CORS for local development and Chrome extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FLAN-T5 (stronger than vanilla T5) with configurable size
# Defaults to 'google/flan-t5-large' for better accuracy; override via env MODEL_NAME
SPEED_MODE = os.getenv("SPEED_MODE", "true").lower() in ("1", "true", "yes", "on")
USE_SMALL_MODEL = os.getenv("USE_SMALL_MODEL", "true").lower() in ("1", "true", "yes", "on")
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")  # e.g., "auto" | "cpu"

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base" if USE_SMALL_MODEL else "google/flan-t5-large")
_loaded_model_name = None
summarizer = None

def _load_pipeline():
    global summarizer, _loaded_model_name
    if summarizer is not None:
        return
    candidates = []
    if MODEL_NAME:
        candidates.append(MODEL_NAME)
    # Fallbacks to lighter models if the chosen one fails to load
    for fallback in ["google/flan-t5-base", "t5-base"]:
        if fallback not in candidates:
            candidates.append(fallback)
    last_error = None
    for name in candidates:
        try:
            model_kwargs = {}
            # Prefer lower precision on accelerators to speed up if available
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # MPS may not fully support fp16, keep default dtype
                pass
            summarizer = pipeline(
                "text2text-generation",
                model=name,
                device_map=DEVICE_MAP,
                model_kwargs=model_kwargs,
                trust_remote_code=False,
            )
            _loaded_model_name = name
            return
        except Exception as e:
            last_error = e
            summarizer = None
            _loaded_model_name = None
            continue
    raise RuntimeError(f"Failed to load any model. Last error: {last_error}")

def _truncate_for_model(text: str, reserve_tokens: int = 64) -> str:
    """Truncate input to fit model's max encoder length, reserving tokens for special tokens.
    Falls back to character-based truncation if tokenizer isn't available yet.
    """
    txt = (text or "").strip()
    if not txt:
        return txt
    try:
        if summarizer is not None and hasattr(summarizer, "tokenizer") and summarizer.tokenizer is not None:
            tok = summarizer.tokenizer
            max_len = getattr(tok, "model_max_length", 512) or 512
            target = max(16, max_len - reserve_tokens)
            # Encode with truncation to target length
            ids = tok.encode(txt, truncation=True, max_length=target)
            return tok.decode(ids, skip_special_tokens=True)
    except Exception:
        pass
    # Fallback: hard cap by characters to roughly match 480 tokens (~ 2000-2400 chars)
    max_chars = 2200
    return txt[:max_chars]

# ---------- ML-based Agents ----------
def simplify(text):
    # Heuristic fallback
    def _fallback_simple(txt: str) -> str:
        txt = (txt or "").strip()
        if not txt:
            return "No text provided."
        parts = [p.strip() for p in txt.replace("\n", " ").split(". ") if p.strip()]
        bullets = parts[:5]
        return "- " + "\n- ".join(bullets)

    try:
        if summarizer is None:
            _load_pipeline()
        input_text = _truncate_for_model(text)
        prompt = (
            "simplify: rewrite in very simple everyday words with short sentences; avoid jargon; "
            "preserve all facts, numbers, and names; do not add anything new. "
            + input_text
        )
        result = summarizer(
            prompt,
            max_new_tokens=96 if SPEED_MODE else 160,
            min_new_tokens=24 if SPEED_MODE else 56,
            do_sample=False,
            num_beams=1 if SPEED_MODE else 8,
            no_repeat_ngram_size=3,
            length_penalty=1.0 if SPEED_MODE else 0.95,
            repetition_penalty=1.12,
            early_stopping=True,
        )
        return result[0]["generated_text"]
    except Exception:
        return _fallback_simple(text)

def paraphrase(text):
    # Heuristic fallback
    def _fallback_para(txt: str) -> str:
        txt = (txt or "").strip()
        if not txt:
            return "No text provided."
        return txt

    try:
        if summarizer is None:
            _load_pipeline()
        input_text = _truncate_for_model(text)
        prompt = (
            "paraphrase: reword while keeping identical meaning; PRESERVE all entities, numbers, units, and key terms; "
            "avoid adding or removing facts; keep style neutral and concise; improve clarity. "
            + input_text
        )
        result = summarizer(
            prompt,
            max_new_tokens=96 if SPEED_MODE else 160,
            min_new_tokens=24 if SPEED_MODE else 56,
            do_sample=False,
            num_beams=1 if SPEED_MODE else 10,
            no_repeat_ngram_size=3,
            length_penalty=1.0 if SPEED_MODE else 0.9,
            repetition_penalty=1.15,
            early_stopping=True,
        )
        return result[0]["generated_text"]
    except Exception:
        return _fallback_para(text)

def ResearchAgent(query):
    try:
        if summarizer is None:
            _load_pipeline()
        prompt = f"summarize research findings for: {_truncate_for_model(query)}"
        result = summarizer(prompt, max_new_tokens=120, min_new_tokens=20, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return "No reliable research summary available."

def TrialsAgent(query):
    try:
        if summarizer is None:
            _load_pipeline()
        prompt = f"summarize clinical trials for: {_truncate_for_model(query)}"
        result = summarizer(prompt, max_new_tokens=120, min_new_tokens=20, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return "No reliable trial summary available."

def PatentAgent(query):
    try:
        if summarizer is None:
            _load_pipeline()
        prompt = f"summarize patents for: {_truncate_for_model(query)}"
        result = summarizer(prompt, max_new_tokens=120, min_new_tokens=20, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return "No reliable patent summary available."

def MarketAgent(query):
    try:
        if summarizer is None:
            _load_pipeline()
        prompt = f"summarize market info for: {_truncate_for_model(query)}"
        result = summarizer(prompt, max_new_tokens=120, min_new_tokens=20, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return "No reliable market summary available."

def NLPAgent(query):
    try:
        if summarizer is None:
            _load_pipeline()
        prompt = f"analyze and explain: {_truncate_for_model(query)}"
        result = summarizer(prompt, max_new_tokens=120, min_new_tokens=20, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return "No reliable NLP analysis available."

# ---------- Master Agent ----------
def MasterAgent(query, mode="simplify"):
    results = {}
    results["simplified"] = simplify(query) if mode=="simplify" else paraphrase(query)
    results["research"] = ResearchAgent(query)
    results["trials"] = TrialsAgent(query)
    results["patents"] = PatentAgent(query)
    results["market"] = MarketAgent(query)
    results["nlp"] = NLPAgent(query)
    return results

# ---------- FastAPI Endpoint ----------
class QueryRequest(BaseModel):
    text: str
    mode: str = "simplify"

@app.post("/process")
async def process(req: QueryRequest):
    try:
        if summarizer is None:
            _load_pipeline()
    except Exception as e:
        # Graceful fallback: return simplified text without model
        return {"output": simplify(req.text)}
    # Fast path: skip multi-agent composition in speed mode
    if SPEED_MODE:
        single = simplify(req.text) if req.mode == "simplify" else paraphrase(req.text)
        return {"output": single}
    sections = MasterAgent(req.text, req.mode)
    # Synthesize a single best output using the T5 model
    combined_input = (
        f"Create a single, clear, helpful explanation. Prioritize correctness and brevity (5-8 sentences).\n"
        f"Original text: {_truncate_for_model(req.text)}\n\n"
        + "\n\n".join([f"{k}: {v}" for k, v in sections.items()])
    )
    best_prompt = (
        "summarize: produce one clear explanation in simple everyday words, short sentences, no jargon; "
        "preserve facts, numbers, and names; do not invent details; paraphrase only to improve clarity. "
        "Output as concise markdown with bullet points; bold important keywords using **like this**; avoid headings. "
        + combined_input
    )
    try:
        best = summarizer(
            best_prompt,
            max_new_tokens=120 if SPEED_MODE else 180,
            min_new_tokens=40 if SPEED_MODE else 70,
            do_sample=False,
            num_beams=4 if SPEED_MODE else 10,
            no_repeat_ngram_size=3,
            length_penalty=0.95,
            repetition_penalty=1.12,
            early_stopping=True,
        )[0]["generated_text"]
        return {"output": best}
    except Exception:
        # Fallback to direct simplify if final synthesis fails
        return {"output": simplify(req.text)}

# ---------- Individual Endpoints ----------
class TextOnlyRequest(BaseModel):
    text: str

@app.post("/simplify")
async def simplify_endpoint(req: TextOnlyRequest):
    try:
        return {"output": simplify(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paraphrase")
async def paraphrase_endpoint(req: TextOnlyRequest):
    try:
        return {"output": paraphrase(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research")
async def research_endpoint(req: TextOnlyRequest):
    try:
        return {"output": ResearchAgent(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trials")
async def trials_endpoint(req: TextOnlyRequest):
    try:
        return {"output": TrialsAgent(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patents")
async def patents_endpoint(req: TextOnlyRequest):
    try:
        return {"output": PatentAgent(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market")
async def market_endpoint(req: TextOnlyRequest):
    try:
        return {"output": MarketAgent(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp")
async def nlp_endpoint(req: TextOnlyRequest):
    try:
        return {"output": NLPAgent(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        if summarizer is None:
            _load_pipeline()
        return {"status": "ok", "model": _loaded_model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unhealthy: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
