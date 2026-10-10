"""Tools to generate responses with the Google Gen AI SDK."""

import logging
import random
import time
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import errors, types


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Create and reuse the client only when Gemini is actually called.

    The old Vertex AI module created its model while the module was imported.
    Lazy creation lets non-Gemini runs import ``llms`` without Google
    credentials. ``genai.Client`` reads either a Gemini API key or the Vertex
    AI environment variables documented in the README.
    """
    return genai.Client()


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 10,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors.APIError as exc:
                if exc.code != 400 or exc.status != "INVALID_ARGUMENT":
                    raise

                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

    return wrapper


@retry_with_exponential_backoff
def generate_from_gemini_completion(
    prompt: list[str | types.Part],
    engine: str,
    max_tokens: int,
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Generate a multimodal response with the model selected by the CLI.

    The ``completion`` name is retained for VisualWebArena compatibility; the
    current SDK sends both text and image parts through ``generate_content``.
    """
    # Preserve VisualWebArena's BLOCK_ONLY_HIGH thresholds for its four
    # concrete harm categories; the old UNSPECIFIED entry was not a rule.
    safety_config = [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_ONLY_HIGH",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]

    generation_config = types.GenerateContentConfig(
        # Match the GPT baseline: use the medim thinking level, fixed seed.
        # removed the previous sampling parameters(temperature, top_p) and limit max_completion_tokens.
        seed=42,
        thinking_config=types.ThinkingConfig(
            thinking_level="medium",
        ),
        safety_settings=safety_config,
    )
    # Request JSON output that follows the action schema.
    if response_schema is not None:
        generation_config.response_mime_type = "application/json"
        generation_config.response_json_schema = response_schema

    # Unlike the old hard-coded gemini-pro-vision instance, this uses the
    # model passed through --model (for example, gemini-3.5-flash-lite).
    response = _get_client().models.generate_content(
        model=engine,
        contents=prompt,
        config=generation_config,
    )
    # Record no-text diagnostics without changing the agent's action path.
    answer = response.text
    if answer is None:
        try:
            candidates = response.candidates or []
            block_reason = (
                response.prompt_feedback.block_reason
                if response.prompt_feedback else None
            )
            logging.getLogger("logger").warning(
                "Gemini returned no text response: id=%s, block_reason=%s, "
                "candidate_count=%d, finish_reasons=%s",
                response.response_id,
                block_reason,
                len(candidates),
                [candidate.finish_reason for candidate in candidates],
            )
        except Exception:
            # Diagnostics must not replace the model's original return value.
            pass
    return answer


# Removed the legacy fake_generate_from_gemini_chat_completion helper: it was
# an unused debug stub, and VisualWebArena's Gemini path supports completion.
