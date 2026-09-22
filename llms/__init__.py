"""This module is adapt from https://github.com/zeno-ml/zeno-build"""
# The google-genai client is created lazily when Gemini is called, so importing
# this module does not require Google Cloud credentials. Avoid a broad except
# here because it would also hide missing dependencies and programming errors.
from .providers.gemini_utils import generate_from_gemini_completion

from .providers.hf_utils import generate_from_huggingface_completion
from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from .utils import call_llm

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "call_llm",
]
