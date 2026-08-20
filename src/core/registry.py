"""
Component Registry: discover & swap modules at runtime.

Mục đích: thay component (encoder, router, reranker, embedder) bằng config 1 dòng,
KHÔNG sửa pipeline code.

Usage:
  from registry import get_component
  encoder = get_component("text_embedder", "online")  # or "offline"
  router = get_component("router", "v7")               # or "v5"
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "router"))


_COMPONENTS = {}


def register(name, version, factory):
    """factory: callable() -> instance."""
    _COMPONENTS[(name, version)] = factory


def get_component(name, version="default", **kwargs):
    """Get instance. Lazy-creates."""
    key = (name, version)
    if key not in _COMPONENTS:
        raise KeyError(f"Component {name}/{version} not registered. "
                       f"Available: {list(_COMPONENTS.keys())}")
    return _COMPONENTS[key](**kwargs) if kwargs else _COMPONENTS[key]()


def list_components():
    return sorted(_COMPONENTS.keys())


# ---- REGISTRATION ----

def _text_embedder_online():
    from offline_fallback import TextEmbedderOnline
    return TextEmbedderOnline()

def _text_embedder_offline():
    from offline_fallback import TextEmbedderOffline
    return TextEmbedderOffline()

def _query_rewriter():
    from query_rewriter import QueryRewriter
    return QueryRewriter()

def _hint_accumulator():
    from hint_accumulator import HintAccumulator
    return HintAccumulator()

def _reranker():
    from reranker import Reranker
    return Reranker()

def _external_image_search():
    from external_image_search import ExternalImageSearch
    return ExternalImageSearch(backend="ddg")


register("text_embedder", "online", _text_embedder_online)
register("text_embedder", "offline", _text_embedder_offline)
register("text_embedder", "default", _text_embedder_online)
register("query_rewriter", "default", _query_rewriter)
register("hint_accumulator", "default", _hint_accumulator)
register("reranker", "default", _reranker)
register("external_image_search", "default", _external_image_search)
register("external_image_search", "ddg", _external_image_search)


if __name__ == "__main__":
    print("=== Component Registry ===\n")
    for k in list_components():
        print(f"  {k[0]}/{k[1]}")
    print(f"\nTotal: {len(_COMPONENTS)} components")
