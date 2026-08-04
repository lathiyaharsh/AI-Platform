from time import time


class ExactCache:
    def __init__(self):
        self.cache = {}
        self.ttl = 300  # seconds

    def get(self, key: str):
        item = self.cache.get(key)

        if not item:
            return None

        value, expires_at = item

        if expires_at < time():
            del self.cache[key]
            return None

        return value

    def set(self, key: str, value: str):
        self.cache[key] = (
            value,
            time() + self.ttl,
        )