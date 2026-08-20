"""Shared TRAKE contracts and event-level multimodal alignment helpers."""

from .contracts import (
    EventEvidence,
    TrakeContractError,
    TrakeEvent,
    normalize_events,
    validate_ranked_sequences,
    validate_sequence_path,
)
from .multimodal import (
    EventLevelMultimodalDante,
    MissingModalityRetriever,
)

__all__ = [
    "EventEvidence",
    "EventLevelMultimodalDante",
    "MissingModalityRetriever",
    "TrakeContractError",
    "TrakeEvent",
    "EventLevelMultimodalDante",
    "normalize_events",
    "validate_ranked_sequences",
    "validate_sequence_path",
]
