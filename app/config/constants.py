from enum import Enum


class Provider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"


class RouteType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    RAG = "rag"
    SUMMARIZE = "summarize"