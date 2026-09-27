from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import json
import os
import chromadb
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"error": "Too many requests. Please slow down."})

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="study_notes")

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_EMBED_MODEL = "gemini-embedding-001"


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------- Embeddings (provider-aware) ----------
async def get_embedding(text: str) -> list[float]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if AI_PROVIDER == "gemini":
                async def _call():
                    r = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}",
                        json={"content": {"parts": [{"text": text}]}},
                    )
                    r.raise_for_status()
                    return r
                response = await call_with_retry(_call)
                return response.json()["embedding"]["values"]
            else:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text},
                )
                response.raise_for_status()
                return response.json()["embedding"]
    except (httpx.HTTPError, httpx.TimeoutException, KeyError) as e:
        raise RuntimeError(f"Embedding request failed: {e}")


async def retrieve_context(question: str, session_id: str, top_k: int = 3) -> list[str]:
    query_embedding = await get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},
    )
    return results["documents"][0] if results["documents"] else []


def build_rag_prompt(question: str, context_chunks: list[str], instructions: str = None) -> str:
    context_text = "\n\n".join(context_chunks) if context_chunks else "No notes available."
    task = instructions or "Answer the student's question using the notes. If the notes don't cover it, say so."
    return f"""You are a helpful study assistant.

Notes:
{context_text}

Task: {task}

Question/Topic: {question}

Important: Give ONLY the final answer. Do not think out loud or narrate your reasoning process. Be direct and concise.

Response:"""


# ---------- Streaming generation (provider-aware) ----------
async def ai_token_stream(prompt: str):
    try:
        if AI_PROVIDER == "gemini":
            async with httpx.AsyncClient(timeout=30.0) as client:
                for attempt in range(3):
                    async with client.stream(
                        "POST",
                        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                    ) as response:
                        if response.status_code in (503, 429) and attempt < 2:
                            print(f"[retry] Gemini attempt {attempt + 1} got {response.status_code}, retrying", flush=True)
                            await asyncio.sleep(2 ** attempt)
                            continue
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                payload = line[len("data: "):].strip()
                                if not payload or payload == "[DONE]":
                                    continue
                                try:
                                    chunk = json.loads(payload)
                                    text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                                    yield {"data": text}
                                except (KeyError, IndexError, json.JSONDecodeError):
                                    continue
                        break
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.1:8b", "prompt": prompt, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            chunk = json.loads(line)
                            yield {"data": chunk.get("response", "")}
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        print(f"[ai_token_stream ERROR - HTTP/Timeout] {e}", flush=True)
        yield {"data": "⚠️ Sorry, the AI service is currently unavailable. Please try again in a moment."}
    except Exception as e:
        print(f"[ai_token_stream ERROR - Unexpected] {type(e).__name__}: {e}", flush=True)
        yield {"data": "⚠️ Something went wrong while generating a response. Please try again."}


async def no_notes_stream():
    yield {"data": "You haven't uploaded any notes yet! Please paste some study notes above and click 'Upload Notes' first, then try again."}


# ---------- Tool definitions ----------
TOOLS_OLLAMA = [
    {"type": "function", "function": {
        "name": "generate_flashcards",
        "description": "Generate study flashcards (question/answer pairs) about a topic",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    }},
    {"type": "function", "function": {
        "name": "generate_quiz",
        "description": "Generate multiple-choice quiz questions about a topic",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string"}, "num_questions": {"type": "integer"}}, "required": ["topic"]},
    }},
    {"type": "function", "function": {
        "name": "summarize_notes",
        "description": "Summarize the stored notes on a topic",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    }},
]

GEMINI_TOOLS = [{"functionDeclarations": [
    {"name": "generate_flashcards", "description": "Generate study flashcards (question/answer pairs) about a topic",
     "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
    {"name": "generate_quiz", "description": "Generate multiple-choice quiz questions about a topic",
     "parameters": {"type": "object", "properties": {
         "topic": {"type": "string"}, "num_questions": {"type": "integer"}}, "required": ["topic"]}},
    {"name": "summarize_notes", "description": "Summarize the stored notes on a topic",
     "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
]}]

ROUTING_INSTRUCTIONS = (
    "You are a routing assistant. Only call a tool/function if the user EXPLICITLY asks for "
    "flashcards, a quiz, or a summary (words like 'flashcards', 'quiz', 'summarize', 'summary'). "
    "For direct questions like 'what is X' or 'explain X', do NOT call any tool."
)


# ---------- Tool decision (provider-aware, normalized return shape) ----------
async def decide_tool(message: str):
    """Returns None, or [{'function': {'name': str, 'arguments': dict}}] regardless of provider."""
    if AI_PROVIDER == "gemini":
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": message}]}],
                    "tools": GEMINI_TOOLS,
                    "systemInstruction": {"parts": [{"text": ROUTING_INSTRUCTIONS}]},
                },
            )
            data = response.json()
            try:
                for part in data["candidates"][0]["content"]["parts"]:
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        return [{"function": {"name": fc["name"], "arguments": fc.get("args", {})}}]
            except (KeyError, IndexError):
                pass
            return None
    else:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [
                        {"role": "system", "content": ROUTING_INSTRUCTIONS},
                        {"role": "user", "content": message},
                    ],
                    "tools": TOOLS_OLLAMA,
                    "stream": False,
                },
            )
            data = response.json()
            return data.get("message", {}).get("tool_calls")


# ---------- Schemas ----------
class NotesInput(BaseModel):
    text: str
    session_id: str


@app.post("/upload-notes")
@limiter.limit("10/minute")
async def upload_notes(request: Request, payload: NotesInput):
    chunks = chunk_text(payload.text)
    ids, embeddings, metadatas = [], [], []
    try:
        for i, chunk in enumerate(chunks):
            embedding = await get_embedding(chunk)
            ids.append(f"{payload.session_id}-chunk-{collection.count() + i}")
            embeddings.append(embedding)
            metadatas.append({"session_id": payload.session_id})
    except RuntimeError:
        return {"error": "Failed to process notes. Please try again in a moment."}

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    session_data = collection.get(where={"session_id": payload.session_id})
    return {"message": "Notes embedded and stored", "total_chunks_stored": len(session_data["ids"])}


@app.delete("/clear-notes")
def clear_notes(session_id: str):
    collection.delete(where={"session_id": session_id})
    return {"message": "Your notes have been cleared"}


@app.get("/chat")
@limiter.limit("15/minute")
async def chat(request: Request, message: str, session_id: str):
    tool_calls = await decide_tool(message)

    if tool_calls:
        call = tool_calls[0]["function"]
        name = call["name"]
        args = call["arguments"]
        topic = args.get("topic", message)

        session_data = collection.get(where={"session_id": session_id})
        if len(session_data["ids"]) == 0:
            return EventSourceResponse(no_notes_stream())

        context_chunks = await retrieve_context(topic, session_id)

        if name == "generate_flashcards":
            instructions = "Generate 5 flashcards as Q: ... A: ... pairs using the notes."
        elif name == "generate_quiz":
            n = args.get("num_questions", 3)
            instructions = f"Generate {n} multiple-choice quiz questions with answers, using the notes."
        elif name == "summarize_notes":
            instructions = "Summarize the notes concisely."
        else:
            instructions = None

        prompt = build_rag_prompt(topic, context_chunks, instructions)
        return EventSourceResponse(ai_token_stream(prompt))

    context_chunks = await retrieve_context(message, session_id)
    prompt = build_rag_prompt(message, context_chunks)
    return EventSourceResponse(ai_token_stream(prompt))

async def call_with_retry(func, *args, max_retries=3, **kwargs):
    """Retries a function on transient errors (503, timeouts) with increasing delay."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (503, 429):  # overloaded or rate-limited
                last_error = e
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"[retry] Attempt {attempt + 1} failed with {e.response.status_code}, retrying in {wait_time}s", flush=True)
                await asyncio.sleep(wait_time)
                continue
            raise
    raise last_error