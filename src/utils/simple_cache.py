"""
Cache layer (in-memory + disk persist) cho query embeds, VLM scores, retrieval results.

API đơn giản — không cần Redis, dùng dict + diskcache fallback.
ROI: query lặp (debug, eval lại, hint progressive) sẽ tránh re-embed/re-VLM.

Idempotent: cache key derived từ (function name, args) qua hash.
"""
import os, json, hashlib, pickle, time
from pathlib import Path
from paths import CACHE_DIR

CACHE_DIR.mkdir(exist_ok=True)


def _key(name: str, *args) -> str:
    """Hash deterministic của (function name, *args)."""
    payload = json.dumps([name, *args], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Cache:
    """Two-tier: in-memory dict + disk pickle. LRU evict on memory."""

    def __init__(self, namespace: str, max_memory: int = 1000):
        self.ns = namespace
        self.mem = {}
        self.access_count = 0
        self.hits = 0
        self.misses = 0
        self.max_memory = max_memory
        self.disk_dir = CACHE_DIR / namespace
        self.disk_dir.mkdir(exist_ok=True)

    def _disk_path(self, key: str) -> Path:
        return self.disk_dir / f"{key}.pkl"

    def get(self, name: str, *args):
        self.access_count += 1
        key = _key(name, *args)
        if key in self.mem:
            self.hits += 1
            return self.mem[key]
        # disk
        fp = self._disk_path(key)
        if fp.exists():
            try:
                v = pickle.load(open(fp, "rb"))
                self._mem_put(key, v)
                self.hits += 1
                return v
            except Exception:
                pass
        self.misses += 1
        return None

    def put(self, value, name: str, *args):
        key = _key(name, *args)
        self._mem_put(key, value)
        try:
            pickle.dump(value, open(self._disk_path(key), "wb"))
        except Exception:
            pass

    def _mem_put(self, key: str, value):
        if len(self.mem) >= self.max_memory:
            # remove first (FIFO simple); LRU not needed for now
            self.mem.pop(next(iter(self.mem)))
        self.mem[key] = value

    def stats(self):
        rate = self.hits / max(1, self.access_count)
        return {
            "namespace": self.ns,
            "access": self.access_count,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": rate,
            "memory_size": len(self.mem),
        }


def cached(namespace: str):
    """Decorator: cache function result by (function name, args).
    Function args must be JSON-serializable (str, int, float, list, dict).
    """
    cache = Cache(namespace)

    def decorator(fn):
        def wrapper(*args, **kwargs):
            # exclude 'self' for instance methods (first arg if class instance)
            cache_args = list(args)
            if cache_args and hasattr(cache_args[0], "__dict__"):
                cache_args = cache_args[1:]  # skip self
            key_args = (*cache_args, *sorted(kwargs.items()))
            cached = cache.get(fn.__name__, *key_args)
            if cached is not None:
                return cached
            result = fn(*args, **kwargs)
            cache.put(result, fn.__name__, *key_args)
            return result
        wrapper._cache = cache
        return wrapper
    return decorator


if __name__ == "__main__":
    # Self-test
    c = Cache("test_smoke")
    assert c.get("foo", 1) is None
    c.put("hello", "foo", 1)
    assert c.get("foo", 1) == "hello"
    assert c.get("foo", 2) is None
    print("Cache self-test PASS")
    print(c.stats())

    # Decorator test
    calls = [0]
    @cached("test_dec")
    def slow_fn(x):
        calls[0] += 1
        return x * 2

    assert slow_fn(5) == 10 and calls[0] == 1
    assert slow_fn(5) == 10 and calls[0] == 1  # cache hit
    assert slow_fn(6) == 12 and calls[0] == 2
    print("Decorator self-test PASS")
    print(slow_fn._cache.stats())
