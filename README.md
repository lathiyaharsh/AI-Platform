# 🚀 Production AI Platform

> **A production-ready AI backend built with FastAPI, LangChain, LlamaIndex, and modern AI engineering principles.**

This repository is my learning journey and long-term project to build an **end-to-end Production AI System** similar to the architecture used by ChatGPT, Cursor, Claude Code, Perplexity, and other enterprise AI applications.

The goal is **not** to build another chatbot, but to build a **scalable AI platform** that demonstrates production-grade architecture, engineering practices, and AI system design.

---

# 🎯 Vision

Build an AI platform capable of:

- Serving multiple AI applications
- Routing requests to different LLM providers
- Managing prompts and versions
- Supporting RAG
- Running AI agents
- Executing tools
- Using memory
- Self-reflecting on responses
- Monitoring performance
- Scaling in production

Every new concept I learn will be integrated into this project.

---

# 🏛 High-Level Architecture

```text
                        Client

                          │

                          ▼

                     FastAPI API

                          │

                    Authentication

                          │

                          ▼

                     AI Gateway

                          │

        Router → Memory → RAG → Context Builder → Prompt Manager

                          │

                   Tool Executor → LLM Providers

                          │

                      Reflection → Cache → Response

        Providers: Groq · Gemini · Future...
```

---

# 📂 Project Structure

```text
app/

├── api/
│   ├── routes/
│   ├── middleware/
│   ├── dependencies/
│   └── schemas/
│
├── gateway/
│   ├── gateway.py
│   ├── router.py
│   ├── decision.py
│   └── providers/
│
├── prompts/
│   ├── manager.py
│   ├── versioning.py
│   ├── registry.py
│   └── templates/
│
├── cache/
│   ├── exact_cache.py
│   ├── semantic_cache.py
│   └── embeddings.py
│
├── memory/
│   ├── conversation.py
│   ├── long_term.py
│   ├── retrieval.py
│   └── context_builder.py
│
├── reflection/
│   ├── __init__.py
│   ├── models.py
│   ├── service.py
│   ├── prompts.py
│   └── types.py
│
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── types.py
│   ├── models.py
│   ├── registry.py
│   ├── executor.py
│   ├── service.py
│   ├── prompts.py
│   └── providers/
│       ├── weather.py
│       ├── calculator.py
│       ├── datetime.py
│       └── uuid_generator.py
│
├── rag/
│   ├── __init__.py
│   ├── service.py
│   ├── models.py
│   ├── types.py
│   ├── constants.py
│   ├── pipeline.py
│   ├── chunking.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── context.py
│   ├── storage.py
│   └── loaders/
│       ├── base.py
│       ├── pdf_loader.py
│       ├── text_loader.py
│       └── markdown_loader.py
│
├── observability/
│   ├── __init__.py
│   ├── service.py
│   ├── models.py
│   ├── types.py
│   ├── metrics.py
│   ├── events.py
│   ├── logger.py
│   ├── collector.py
│   ├── cost.py
│   └── tracing.py
│
├── background/
│   ├── jobs.py
│   ├── workers.py
│   └── scheduler.py
│
├── llm/
│
├── services/
│
├── database/
│
├── config/
│
└── main.py
```

---

# 🧠 Core Components

## ✅ AI Gateway

Responsible for:

- Single entry point
- Provider abstraction
- Retry
- Fallback
- Request orchestration

---

## ✅ Intelligent Router

Automatically decides:

- Which provider to use
- Which model to use
- Coding vs Chat vs RAG
- Fast vs Smart model

---

## ✅ Prompt Manager

Handles:

- Prompt templates
- System prompts
- Prompt registry
- Variables

---

## ✅ Prompt Versioning

Supports:

- Version history
- Rollback
- Experimentation
- Prompt updates

---

## ✅ Exact Cache

Caches identical requests.

---

## ✅ Semantic Cache

Uses embeddings to reuse answers from similar prompts.

---

## ✅ Memory

Supports:

- Conversation Memory
- Long-Term Memory
- Context Retrieval

---

## ✅ Context Builder

Creates optimized prompts using:

- Chat history
- Memory
- Retrieved documents
- System instructions

---

## ✅ Reflection Engine

Allows the AI to:

- Review responses
- Find mistakes
- Improve answers
- Self-correct

---

## ✅ Tool Execution

Supports:

- Tool registry
- Tool executor
- Tool service (Gateway-facing API)
- JSON schemas for LLM function calling
- Multi-tool / multi-round tool loop
- Validation
- Parallel execution
- Timeouts
- Built-in tools: weather, calculator, datetime, uuid_generator

---

## ✅ RAG

Features:

- Document ingestion
- Chunking
- Embeddings
- Vector Search
- Retrieval

---

## ✅ Observability

`ObservabilityService` (injected into Gateway) traces every AI request:

- Request ID on every structured JSON log line
- Latency, provider, model, route, prompt version
- Token usage (provider usage when available, else estimated)
- CostEstimator (Groq / Gemini; pricing table is hot-updatable)
- Exact / semantic cache hit rates
- Router, memory, RAG, tool, and reflection metrics
- Retries, fallbacks, errors

Pluggable `MetricsCollector` exporters — register OpenTelemetry, Prometheus,
Langfuse, Helicone, Phoenix, or W&B later without Gateway changes.

Config:

```text
OBSERVABILITY_ENABLED=true
ESTIMATE_COST=true
LOG_REQUEST_BODY=false
LOG_RESPONSE_BODY=false
ENABLE_JSON_LOGGING=true
```

---

## ✅ Background Jobs

Used for:

- PDF indexing
- Embedding generation
- Long-running AI tasks
- Batch processing

---

# 🧩 Technology Stack

## Backend

- FastAPI
- Python

## AI

- LangChain
- LlamaIndex

## Providers

- Groq
- Google Gemini

## Embeddings

- Hugging Face Inference API

## Database

- Supabase PostgreSQL
- pgvector

## Future Infrastructure

- Redis
- Docker
- Kubernetes

---

# 🔐 Environment Variables

```env
# App

APP_NAME=Production AI Platform
APP_ENV=development

# Local only — do NOT set on FastAPI Cloud (PORT/HOST are reserved / invalid there).
# HOST=0.0.0.0
# PORT=8000

LOG_LEVEL=INFO

# Providers

GROQ_API_KEY=

GEMINI_API_KEY=

HUGGINGFACE_API_KEY=

# Database

SUPABASE_DB_URL=

SUPABASE_COLLECTION=ai_chat_docs

HF_EMBED_DIM=384

# Default Settings

DEFAULT_PROVIDER=groq
DEFAULT_MODEL=openai/gpt-oss-120b

CACHE_ENABLED=true
SEMANTIC_CACHE=true

REFLECTION_ENABLED=true
REFLECTION_PROVIDER=groq
REFLECTION_MODEL=openai/gpt-oss-120b
REFLECTION_THRESHOLD=0.7
MAX_REFLECTIONS=1

TOOLS_ENABLED=true
TOOLS_PROVIDER=groq
TOOLS_MODEL=openai/gpt-oss-120b
TOOL_TIMEOUT_SECONDS=15
MAX_TOOL_ITERATIONS=3

# RAG

ENABLE_RAG=true
TOP_K=5
MIN_SIMILARITY=0.55
CHUNK_SIZE=512
CHUNK_OVERLAP=64
CHUNK_STRATEGY=sentence
VECTOR_STORE=pgvector

# Observability

OBSERVABILITY_ENABLED=true
ESTIMATE_COST=true
LOG_REQUEST_BODY=false
LOG_RESPONSE_BODY=false
ENABLE_JSON_LOGGING=true
```

---

# 📚 Learning Journey

## ✅ Backend Engineering

- FastAPI
- SQLModel
- Authentication
- Middleware
- Dependencies
- Background Tasks
- Streaming APIs

---

## ✅ LLM Fundamentals

- Prompt Engineering
- Tokens
- Context Window
- Temperature
- Tool Calling

---

## ✅ LangChain

- Chains
- Memory
- Tools
- Agents

---

## ✅ LlamaIndex

- RAG
- Document Indexing
- Retrieval
- Chunking
- Embeddings

---

## ✅ AI System Design

- Gateway
- Intelligent Routing
- Prompt Management
- Prompt Versioning
- Exact Cache
- Semantic Cache
- Retry Logic
- Fallback Providers
- Observability
- Background Jobs

---

## ✅ Advanced AI Agents

- Planning
- ReAct Loop
- Memory Architecture
- Context Builder
- Reflection
- Tool Execution

---

## ✅ MCP (Model Context Protocol)

- Fundamentals
- MCP Server Architecture

---

## ✅ Multi-Agent Systems

- Coordinator Agent
- Specialized Agents
- Shared Memory
- Parallel Execution

---

## ✅ Production Infrastructure

- Docker Concepts
- Kubernetes Concepts
- Load Balancing
- Worker Architecture
- Scaling Fundamentals

---

# 🚧 Roadmap

> **Current progress:** Phases 1–8 core path is live (`POST /chat` / `POST /chat/rag` → router → long-term memory → RAG retrieve → context builder → versioned prompts → exact/semantic cache → tool loop → Groq/Gemini with retry/fallback → reflection → conversation memory + document catalog → ObservabilityService request summary). Vector persistence uses Supabase Postgres + pgvector (`VECTOR_STORE=pgvector`); Redis and multi-agent work are still ahead.

## Phase 1 — Production AI Gateway

- [x] Gateway
- [x] Router
- [x] Providers (Groq, Gemini)
- [x] Retry
- [x] Fallback

---

## Phase 2 — Prompt Management

- [x] Prompt Manager
- [x] Prompt Versioning (file-based history, active version, rollback)
- [x] Prompt Registry

---

## Phase 3 — Memory

- [x] Conversation Memory (in-memory store + sliding window)
- [x] Long-Term Memory (in-memory facts + explicit extract, survives window)
- [x] Context Builder (history + memory/docs hooks → versioned prompt)

---

## Phase 4 — Caching

- [x] Exact Cache (in-memory + TTL)
- [x] Semantic Cache (in-memory + HuggingFace embeddings)

---

## Phase 5 — Reflection

- [x] Reflection Engine
- [x] Auto Improvement

---

## Phase 6 — Tool Execution

- [x] Tool Registry
- [x] Tool Executor
- [x] Tool Service + multi-round tool loop
- [x] Sample tools (weather, calculator, datetime, uuid)

---

## Phase 7 — RAG

- [x] Document Pipeline (txt / md / pdf → clean → chunk → embed → store)
- [x] Retrieval (EmbeddingService + VectorStore + Reranker)
- [x] Vector Search (MemoryVectorStore + PgVectorStore / cosine similarity)
- [x] Context attribution + Gateway / Router / API integration

---

## Phase 8 — Observability

- [x] ObservabilityService (request lifecycle + summary)
- [x] Structured JSON logging (request_id bound)
- [x] Metrics (latency, cache, tools, RAG, reflection, memory, router)
- [x] Token usage + CostEstimator (Groq / Gemini)
- [x] Pluggable collector (OTel / Prometheus / Langfuse / Helicone / Phoenix ready)
- [x] Tracing bridge (exporter hooks without Gateway changes)

---

# 🎯 Long-Term Goal

This repository will evolve into a complete **Production AI Platform** showcasing modern AI engineering concepts, including:

- AI Gateway
- Agent Runtime
- RAG
- MCP
- Multi-Agent Systems
- LLMOps
- AI Evaluation
- Guardrails
- Observability
- Production Deployment

Rather than creating multiple disconnected demo projects, this repository will continuously grow into a real-world AI platform that demonstrates both architectural design and practical engineering.