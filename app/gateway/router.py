from app.config.constants import Provider, RouteType
from app.gateway.decision import Decision


class Router:
    """
    Intelligent request router.
    """

    def route(self, prompt: str) -> Decision:

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