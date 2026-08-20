"""Experimental, dependency-injected VQA tools (not the production pipeline)."""

from .answer_provider import (
    AnswerProvider,
    AnswerProviderConfigurationError,
    AnswerProviderError,
    AnswerProviderInputError,
    AnswerProviderRequest,
    AnswerProviderRequestError,
    AnswerProviderResponse,
    AnswerProviderSchemaError,
    EvidenceBundle,
    FrameEvidence,
    OpenAICompatibleAnswerProvider,
    QwenLocalAnswerProvider,
    RetryPolicy,
)

from .tools import (
    ASRContextTool, CalculatorTool, OCRTool, QuestionType, ToolResult,
    ToolContext, VLMTool,
)
from .orchestrator import EvidenceAnswer, ParallelVQAOrchestrator

__all__ = [
    "AnswerProvider", "AnswerProviderConfigurationError", "AnswerProviderError",
    "AnswerProviderInputError", "AnswerProviderRequest", "AnswerProviderRequestError",
    "AnswerProviderResponse", "AnswerProviderSchemaError", "EvidenceBundle",
    "FrameEvidence", "OpenAICompatibleAnswerProvider", "QwenLocalAnswerProvider",
    "RetryPolicy",
    "ASRContextTool", "CalculatorTool", "OCRTool", "QuestionType",
    "ToolResult", "ToolContext", "VLMTool", "EvidenceAnswer",
    "ParallelVQAOrchestrator",
]
