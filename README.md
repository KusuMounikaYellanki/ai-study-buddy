# 📚 AI Study Buddy

An AI-powered study assistant that answers questions from your own notes using Retrieval-Augmented Generation (RAG), with agentic tool-calling for flashcards, quizzes, and summaries — all streamed in real time.

**Live demo:** _[add your deployed URL here once live]_

## What it does

- **Upload your notes** — text gets chunked and embedded into a vector database
- **Ask questions** — answers are grounded in your notes, not generic AI knowledge, and the app honestly says so when something isn't covered
- **Real-time streaming** — responses appear token-by-token, like ChatGPT
- **Agentic tool-calling** — the AI itself decides whether to answer directly, generate flashcards, build a quiz, or summarize, based on what you ask
- **Multi-tenant** — every user's notes and chat history are private and isolated from other users
- **Multi-provider AI** — runs on a local open-weight model (Ollama) for development, or a hosted model (Google Gemini) for deployment, via one config switch

## Architecture

- **Backend:** FastAPI (Python), async streaming via Server-Sent Events
- **Vector database:** ChromaDB, session-scoped for user isolation
- **AI providers:** Ollama (local, free) or Google Gemini API (hosted, free tier)
- **Frontend:** Vanilla HTML/CSS/JS, no framework — chat UI with streaming, chat history, and a notes upload panel
- **Rate limiting:** slowapi, capped per IP to protect API usage

## Running it locally

1. Clone this repo and create a virtual environment:
```bash
   git clone https://github.com/KusuMounikaYellanki/ai-study-buddy.git
   cd ai-study-buddy
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file:
    AI_PROVIDER=ollama
    GEMINI_API_KEY=your-gemini-key-here

4. For local dev with Ollama, install [Ollama](https://ollama.com) and pull the models:
```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
```

5. Run the server:
```bash
   uvicorn main:app --reload --port 8002
```

6. Open `http://127.0.0.1:8002` in your browser.

## Why I built this

Built to learn and demonstrate practical Gen AI engineering: RAG pipeline design, real-time streaming architecture, agentic tool-calling, multi-tenant data isolation, and provider-agnostic AI integration.
