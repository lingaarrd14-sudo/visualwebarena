import base64
import json
import os
from io import BytesIO
from typing import List
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# Latest model revision available before the original VisualWebArena release.
BLIP2_REVISION = "e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77"
REMOTE_CAPTION_MAX_BODY_BYTES = 64 * 1024 * 1024
REMOTE_CAPTION_BATCH_SIZE = 2


def get_captioning_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    if "blip2" in model_name:
        captioning_processor = Blip2Processor.from_pretrained(
            model_name, revision=BLIP2_REVISION
        )
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            revision=BLIP2_REVISION,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )
    captioning_model.to(device)

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        if prompt is None:
            # Perform VQA
            inputs = captioning_processor(
                images=images, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            # Regular captioning. Prompt is a list of strings, one for each image
            assert len(images) == len(
                prompt
            ), "Number of images and prompts must match, got {} and {}".format(
                len(images), len(prompt)
            )
            inputs = captioning_processor(
                images=images, text=prompt, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return captions

    return caption_images


def get_remote_captioning_fn(base_url: str, model_name: str) -> callable:
    """Use a BLIP-2 server through a local SSH forward."""
    parsed_url = urlparse(base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in {
        "127.0.0.1", "localhost", "::1"
    }:
        raise ValueError("VWA_CAPTIONER_URL must be an HTTP loopback address")
    base_url = base_url.rstrip("/")

    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        health = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"BLIP-2 server is unavailable at {base_url}; check the SSH tunnel"
        ) from exc
    if not isinstance(health, dict):
        raise RuntimeError("BLIP-2 server returned invalid health data")
    if (
        health.get("model") != model_name
        or health.get("revision") != BLIP2_REVISION
        or not health.get("cuda")
    ):
        raise RuntimeError(
            f"BLIP-2 server has model={health.get('model')!r}, "
            f"revision={health.get('revision')!r}, cuda={health.get('cuda')!r}; "
            f"expected model={model_name!r}, revision={BLIP2_REVISION!r} on CUDA"
        )

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        if prompt is not None and len(prompt) != len(images):
            raise ValueError("Number of images and prompts must match")

        encoded_images = []
        for image in images:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

        captions = []
        start = 0
        while start < len(encoded_images):
            end = min(start + REMOTE_CAPTION_BATCH_SIZE, len(encoded_images))
            while True:
                request_body = json.dumps({
                    "model": model_name,
                    "images": encoded_images[start:end],
                    "prompt": prompt[start:end] if prompt is not None else None,
                    "max_new_tokens": max_new_tokens,
                }).encode("utf-8")
                if len(request_body) <= REMOTE_CAPTION_MAX_BODY_BYTES:
                    break
                if end - start == 1:
                    raise RuntimeError(
                        f"Image {start} exceeds the BLIP-2 request size limit"
                    )
                end -= 1

            try:
                response = requests.post(
                    f"{base_url}/caption",
                    data=request_body,
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
            except requests.HTTPError as exc:
                detail = response.text[:500]
                raise RuntimeError(
                    f"Remote BLIP-2 caption request failed: {detail}"
                ) from exc
            except (requests.RequestException, ValueError) as exc:
                raise RuntimeError("Remote BLIP-2 caption request failed") from exc
            if not isinstance(result, dict):
                raise RuntimeError("Remote BLIP-2 server returned invalid JSON")
            batch_captions = result.get("captions")
            if (
                not isinstance(batch_captions, list)
                or len(batch_captions) != end - start
                or not all(isinstance(caption, str) for caption in batch_captions)
            ):
                raise RuntimeError("Remote BLIP-2 server returned invalid captions")
            captions.extend(batch_captions)
            start = end
        return captions

    return caption_images


def get_configured_captioning_fn(device, dtype, model_name: str) -> callable:
    """Choose the remote GPU when configured, otherwise preserve local behavior."""
    remote_url = os.environ.get("VWA_CAPTIONER_URL")
    if remote_url:
        return get_remote_captioning_fn(remote_url, model_name)
    return get_captioning_fn(device, dtype, model_name)


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
