"""
Module 1 (L10): Persistent cache layer cho query embeddings + retrieval results.

Mục tiêu: < 500ms latency end-to-end cho repeated queries.
- query_embed (bge-m3 + sigLIP-text): cache theo query string
- retrieval_top_k (final result): cache theo (query, pipeline_version)

Dùng diskcache (SQLite-backed) — persist qua restart, no server.
Key strategy: SHA256 hash + version namespace.

Self-test: cache hit < 1ms vs API ~200ms.
"""
import os, hashlib, time, pickle
from functools import wraps
import numpy as np

try:
    import diskcache as dc
except ImportError:
    dc = None
    print("[Cache] diskcache không có, dùng dict in-memory")

# Resolve cache relative to the active repository, including after a move.
try:
    from paths import CACHE_DIR
    CACHE_DIR = str(CACHE_DIR)
except Exception:
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "data", "cache")
CACHE_DIR = os.path.abspath(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)


class Cache:
    """Persistent cache, namespaced by version."""

    def __init__(self, namespace="default", size_limit_gb=2, version="v1"):
        self.namespace = namespace
        self.version = version
        if dc:
            self._dc = dc.Cache(os.path.join(CACHE_DIR, namespace),
                                size_limit=size_limit_gb * 1024**3)
        else:
            self._dc = {}

    def _key(self, *args):
        h = hashlib.sha256(repr((self.version, args)).encode()).hexdigest()[:32]
        return h

    def get(self, *args):
        k = self._key(*args)
        if dc:
            return self._dc.get(k)
        return self._dc.get(k)

    def set(self, value, *args):
        k = self._key(*args)
        if dc:
            self._dc.set(k, value)
        else:
            self._dc[k] = value

    def cached(self, fn):
        """Decorator: cache fn(*args) → result.
        QUAN TRỌNG: nếu fn là method (có self), bỏ self khỏi key (object id != value).
        """
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Bỏ qua self (instance) nếu là method
            key_args = args
            if args and hasattr(args[0], "__class__") and not isinstance(args[0], (str, int, float, tuple, list, dict)):
                key_args = args[1:]
            key_args = key_args + tuple(sorted(kwargs.items()))
            v = self.get(fn.__name__, *key_args)
            if v is not None:
                return v
            v = fn(*args, **kwargs)
            self.set(v, fn.__name__, *key_args)
            return v
        return wrapper

    def clear(self):
        if dc: self._dc.clear()
        else: self._dc.clear()

    def stats(self):
        if dc:
            return {"size": len(self._dc), "volume": self._dc.volume()}
        return {"size": len(self._dc)}


# Pre-instantiated caches cho từng namespace
_caches = {}
def get_cache(namespace, version="v1"):
    key = (namespace, version)
    if key not in _caches:
        _caches[key] = Cache(namespace=namespace, version=version)
    return _caches[key]


if __name__ == "__main__":
    print("=== Cache self-test ===")
    c = get_cache("test", version="v1")
    c.clear()

    # Test 1: basic get/set
    c.set("hello world", "key1")
    assert c.get("key1") == "hello world"
    print("  ✓ basic get/set")

    # Test 2: numpy roundtrip
    v = np.random.randn(100).astype(np.float32)
    c.set(v, "np_key")
    v2 = c.get("np_key")
    assert np.allclose(v, v2)
    print("  ✓ numpy roundtrip")

    # Test 3: decorator
    call_count = [0]
    @c.cached
    def slow(x):
        call_count[0] += 1
        time.sleep(0.05)
        return x * 2
    t0 = time.time(); a = slow(5); t1 = time.time()-t0
    t0 = time.time(); b = slow(5); t2 = time.time()-t0
    assert a == b == 10
    assert call_count[0] == 1, f"expected 1 call, got {call_count[0]}"
    print(f"  ✓ decorator: 1st {t1*1000:.0f}ms, 2nd {t2*1000:.2f}ms "
          f"(speedup ~{t1/(t2 if t2>1e-6 else 1e-6):.0f}x)")

    # Test 4: persistence (write+read different cache instance)
    c.set("persist_value", "persist_key")
    c2 = Cache(namespace="test", version="v1")
    assert c2.get("persist_key") == "persist_value"
    print("  ✓ persistence")

    print(f"\n  stats: {c.stats()}")
    c.clear()
    print("\n=== ALL PASS ===")
