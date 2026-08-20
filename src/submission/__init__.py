"""Provider-agnostic submission contracts for Q&A and TRAKE."""

from .adapters import (
    serialize_qna_submission,
    serialize_trake_submission,
    serialize_submission,
)
from .contracts import (
    QnAAnswerRecord,
    TrakeAnswerRecord,
    canonical_frame_index,
)
from .validators import (
    validate_qna_answers,
    validate_trake_answers,
)

__all__ = [
    "QnAAnswerRecord",
    "TrakeAnswerRecord",
    "canonical_frame_index",
    "serialize_qna_submission",
    "serialize_trake_submission",
    "serialize_submission",
    "validate_qna_answers",
    "validate_trake_answers",
]
