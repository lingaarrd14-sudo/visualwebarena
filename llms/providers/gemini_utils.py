"""Tools to generate responses with the Google Gen AI SDK."""

import random
import time
from functools import lru_cache

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
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
):
    """Retry only rate limits and transient server failures.

    Invalid requests are raised immediately because retrying them with the
    same prompt and configuration cannot succeed. The base-2 backoff and
    three-retry default match the OpenAI provider's retry policy.
    """

    def wrapper(*args, **kwargs):  # type: ignore
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors.APIError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise

                num_retries += 1
                if num_retries > max_retries:
                    raise RuntimeError(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    ) from exc

                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

    return wrapper


@retry_with_exponential_backoff
def generate_from_gemini_completion(
    prompt: list[str | types.Part],
    engine: str,
    max_tokens: int,
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
        # Match the GPT baseline: use the same output budget, fixed seed, and
        # medium reasoning. Sampling parameters are intentionally omitted.
        max_output_tokens=max_tokens,
        seed=42,
        thinking_config=types.ThinkingConfig(
            thinking_level="medium",
        ),
        safety_settings=safety_config,
    )

    # Unlike the old hard-coded gemini-pro-vision instance, this uses the
    # model passed through --model (for example, gemini-3.8-flash).
    response = _get_client().models.generate_content(
        model=engine,
        contents=prompt,
        config=generation_config,
    )
    # response.text can be absent when Gemini returns no text candidate.
    if response.text is None:
        raise RuntimeError("Gemini returned no text response.")
    return response.text


# Removed the legacy fake_generate_from_gemini_chat_completion helper: it was
# an unused debug stub, and VisualWebArena's Gemini path supports completion.
