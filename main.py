from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import httpx
import json
import chromadb

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="study_notes")


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


# ---------- Embeddings ----------
async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        return response.json()["embedding"]


# ---------- Retrieval, now scoped to one session ----------
async def retrieve_context(question: str, session_id: str, top_k: int = 3) -> list[str]:
    query_embedding = await get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},   # <-- the isolation boundary
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

Important: Give ONLY the final answer. Do not think out loud, do not explain your reasoning process, and do not mention that you are reconsidering or changing your answer. Be direct and concise.

Response:"""


# ---------- Streaming generation ----------
async def ollama_token_stream(prompt: str):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "http://localhost:11434/api/generate",
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    yield {"data": chunk.get("response", "")}


async def no_notes_stream():
    yield {"data": "You haven't uploaded any notes yet! Please paste some study notes above and click 'Upload Notes' first, then try again."}


# ---------- Tool definitions ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_flashcards",
            "description": "Generate study flashcards (question/answer pairs) about a topic",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "The topic to make flashcards about"}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": "Generate multiple-choice quiz questions about a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to quiz on"},
                    "num_questions": {"type": "integer", "description": "How many questions"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_notes",
            "description": "Summarize the stored notes on a topic",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "The topic to summarize"}},
                "required": ["topic"],
            },
        },
    },
]


async def decide_tool(message: str):
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a routing assistant. Only call a tool if the user "
                            "EXPLICITLY asks for flashcards, a quiz, or a summary "
                            "(using words like 'flashcards', 'quiz', 'summarize', 'summary'). "
                            "For direct questions like 'what is X', 'explain X', or 'how does X work', "
                            "do NOT call any tool — these should be answered normally."
                        ),
                    },
                    {"role": "user", "content": message},
                ],
                "tools": TOOLS,
                "stream": False,
            },
        )
        data = response.json()
        return data.get("message", {}).get("tool_calls")


# ---------- Schemas ----------
class NotesInput(BaseModel):
    text: str
    session_id: str


# ---------- Ingestion, now tagged with session_id ----------
@app.post("/upload-notes")
async def upload_notes(payload: NotesInput):
    chunks = chunk_text(payload.text)
    ids, embeddings, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        embedding = await get_embedding(chunk)
        # Unique ID per chunk, safe even across sessions
        ids.append(f"{payload.session_id}-chunk-{collection.count() + i}")
        embeddings.append(embedding)
        metadatas.append({"session_id": payload.session_id})

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    # Count only this session's chunks for an accurate per-user total
    session_count = collection.get(where={"session_id": payload.session_id})
    total_for_session = len(session_count["ids"])

    return {"message": "Notes embedded and stored", "total_chunks_stored": total_for_session}


# ---------- Clear notes, scoped to one session only ----------
@app.delete("/clear-notes")
def clear_notes(session_id: str):
    collection.delete(where={"session_id": session_id})
    return {"message": "Your notes have been cleared"}


# ---------- Agentic chat endpoint, now session-aware ----------
@app.get("/chat")
async def chat(message: str, session_id: str):
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
        return EventSourceResponse(ollama_token_stream(prompt))

    context_chunks = await retrieve_context(message, session_id)
    prompt = build_rag_prompt(message, context_chunks)
    return EventSourceResponse(ollama_token_stream(prompt))

