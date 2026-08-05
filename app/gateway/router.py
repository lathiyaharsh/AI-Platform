from app.config.constants import Provider, RouteType
from app.config.settings import settings
from app.gateway.decision import Decision


class Router:
    """
    Intelligent request router.

    Routes: CHAT | CODE | SUMMARY | RAG

    Keyword heuristics today; architecture supports future
    AI intent classifiers without changing Gateway.
    """

    RAG_KEYWORDS = (
        "document",
        "documents",
        "according to",
        "based on the",
        "from the file",
        "from the pdf",
        "in the docs",
        "in the document",
        "knowledge base",
        "uploaded",
        "retrieve",
        "citation",
        "source attribution",
        "what does the doc",
        "what does the document",
        "rag",
    )

    def route(
        self,
        prompt: str,
        *,
        force_rag: bool = False,
        rag_available: bool = False,
    ) -> Decision:
        if force_rag or (
            settings.enable_rag
            and rag_available
            and self._is_rag(prompt)
        ):
            return Decision(
                provider=Provider.GROQ,
                route=RouteType.RAG,
                model="llama-3.3-70b-versatile",
            )

        text = prompt.lower()

        if any(word in text for word in [
            "python",
            "javascript",
            "fastapi",
            "react",
            "code",
            "bug",
            "function",
            "sql",
        ]):
            return Decision(
                provider=Provider.GROQ,
                route=RouteType.CODE,
                model="llama-3.3-70b-versatile",
            )

        if any(word in text for word in [
            "summarize",
            "summary",
        ]):
            return Decision(
                provider=Provider.GEMINI,
                route=RouteType.SUMMARIZE,
                model="gemini-2.5-flash",
            )

        return Decision(
            provider=Provider.GROQ,
            route=RouteType.CHAT,
            model="llama-3.3-70b-versatile",
        )

    def _is_rag(self, prompt: str) -> bool:
        text = prompt.lower()
        return any(keyword in text for keyword in self.RAG_KEYWORDS)
