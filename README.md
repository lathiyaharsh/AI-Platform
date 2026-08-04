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

        ┌──────────────┬──────────────┐

        ▼              ▼              ▼

     Router      Prompt Manager    Memory

        │              │              │

        └──────────────┼──────────────┘

                       ▼

                Context Builder

                       ▼

               Reflection Engine

                       ▼

                Tool Executor

                       ▼

                 RAG / Retrieval

                       ▼

                LLM Providers

        ┌─────────┬─────────┬─────────┐

        ▼         ▼         ▼

      Groq      Gemini    Future...
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
│   ├── engine.py
│   ├── evaluator.py
│   └── prompts.py
│
├── tools/
│   ├── registry.py
│   ├── executor.py
│   └── implementations/
│
├── rag/
│   ├── ingestion.py
│   ├── indexing.py
│   ├── retrieval.py
│   └── chunking.py
│
├── observability/
│   ├── logger.py
│   ├── metrics.py
│   ├── tracing.py
│   └── cost.py
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
- Tool execution
- Validation
- Parallel execution
- Timeouts

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

Tracks:

- Latency
- Token usage
- Cost
- Cache hits
- Retries
- Errors
- Provider performance

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
# Providers

GROQ_API_KEY=

GEMINI_API_KEY=

HUGGINGFACE_API_KEY=

# Database

SUPABASE_DB_URL=

# Default Settings

DEFAULT_PROVIDER=groq
DEFAULT_MODEL=llama-3.3-70b-versatile

CACHE_ENABLED=true
SEMANTIC_CACHE=true
REFLECTION_ENABLED=true

LOG_LEVEL=INFO
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

> **Current progress:** Phases 1–4 core path is live (`POST /chat` → router → prompts → exact/semantic cache → Groq/Gemini with retry/fallback → in-memory conversation memory). Persistence (Postgres/Redis), RAG, agents, reflection, tools, and deployment are still ahead.

## Phase 1 — Production AI Gateway

- [x] Gateway
- [x] Router
- [x] Providers (Groq, Gemini)
- [x] Retry
- [x] Fallback

---

## Phase 2 — Prompt Management

- [x] Prompt Manager
- [ ] Prompt Versioning
- [x] Prompt Registry

---

## Phase 3 — Memory

- [x] Conversation Memory (in-memory store + sliding window)
- [ ] Long-Term Memory
- [ ] Context Builder

---

## Phase 4 — Caching

- [x] Exact Cache (in-memory + TTL)
- [x] Semantic Cache (in-memory + HuggingFace embeddings)

---

## Phase 5 — Reflection

- [ ] Reflection Engine
- [ ] Auto Improvement

---

## Phase 6 — Tool Execution

- [ ] Tool Registry
- [ ] Tool Executor

---

## Phase 7 — RAG

- [ ] Document Pipeline
- [ ] Retrieval
- [ ] Vector Search

---

## Phase 8 — Observability

- [x] Logging (loguru + request middleware)
- [ ] Metrics
- [ ] Cost Tracking
- [ ] Tracing

---

## Phase 9 — LLMOps

- [ ] Prompt Evaluation
- [ ] Prompt Testing
- [ ] Hallucination Detection
- [ ] A/B Testing
- [ ] Golden Dataset

---

## Phase 10 — AI Safety

- [ ] Prompt Injection Detection
- [ ] Output Validation
- [ ] PII Detection
- [ ] Guardrails

---

## Phase 11 — Production Deployment

- [ ] Redis (replace in-memory cache / memory backends)
- [ ] Docker
- [ ] Kubernetes
- [ ] CI/CD
- [ ] Monitoring

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