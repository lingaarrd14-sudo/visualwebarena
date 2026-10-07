import argparse
from typing import Any

# The google-genai SDK represents multimodal inputs with Part instead of the
# legacy vertexai.preview.generative_models.Image class.
from google.genai import types as genai_types

from llms import (
    generate_from_gemini_completion,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    # Pass the schema only when JSON output is requested.
    schema_kwargs = {}
    if response_schema is not None:
        schema_kwargs["response_schema"] = response_schema
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
                **schema_kwargs,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, genai_types.Part) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            max_tokens=lm_config.gen_config["max_tokens"],
            **schema_kwargs,
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
