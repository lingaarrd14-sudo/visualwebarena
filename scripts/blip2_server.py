"""Serve BLIP-2 to a local runner through an SSH port forward.

This server only listens on the Pod's loopback interface. Start it on a CUDA
machine, then forward its port with SSH from the benchmark machine.
"""

import argparse
import base64
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Callable

from PIL import Image

MAX_BODY_BYTES = 64 * 1024 * 1024
# Latest model revision available before the original VisualWebArena release.
BLIP2_REVISION = "e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77"


def make_handler(caption_fn: Callable, model_name: str, gpu_name: str):
    class CaptionHandler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(
                200,
                {
                    "model": model_name,
                    "revision": BLIP2_REVISION,
                    "cuda": True,
                    "gpu": gpu_name,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/caption":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("request body is empty or too large")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if payload.get("model") != model_name:
                    raise ValueError("model does not match the loaded BLIP-2 model")
                encoded_images = payload.get("images")
                if not isinstance(encoded_images, list) or not 0 < len(encoded_images) <= 64:
                    raise ValueError("images must contain 1 to 64 entries")
                prompt = payload.get("prompt")
                if prompt is not None and (
                    not isinstance(prompt, list)
                    or len(prompt) != len(encoded_images)
                    or not all(isinstance(item, str) for item in prompt)
                ):
                    raise ValueError("prompt must contain one string per image")
                max_new_tokens = payload.get("max_new_tokens", 32)
                if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
                    raise ValueError("max_new_tokens must be between 1 and 256")

                images = []
                for encoded_image in encoded_images:
                    if not isinstance(encoded_image, str):
                        raise ValueError("every image must be base64 text")
                    with Image.open(BytesIO(base64.b64decode(encoded_image, validate=True))) as image:
                        images.append(image.convert("RGB"))
                captions = caption_fn(images, prompt, max_new_tokens)
                if len(captions) != len(images):
                    raise RuntimeError("BLIP-2 returned the wrong number of captions")
                self.send_json(200, {"captions": captions})
            except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                self.send_json(500, {"error": "BLIP-2 inference failed; check Pod logs"})

    return CaptionHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Loopback-only BLIP-2 caption server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="Salesforce/blip2-flan-t5-xl")
    args = parser.parse_args()

    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; start this server on a GPU Pod")
    gpu_name = torch.cuda.get_device_name(0)
    if "blip2" not in args.model.lower():
        parser.error("Only BLIP-2 models are supported")
    processor = Blip2Processor.from_pretrained(
        args.model, revision=BLIP2_REVISION
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        revision=BLIP2_REVISION,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    def caption(images, prompt, max_new_tokens):
        with torch.inference_mode():
            if prompt is None:
                inputs = processor(images=images, return_tensors="pt")
            else:
                inputs = processor(images=images, text=prompt, return_tensors="pt")
            inputs = inputs.to("cuda", torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)

    server = HTTPServer(("127.0.0.1", args.port), make_handler(caption, args.model, gpu_name))
    print(f"BLIP-2 ready on 127.0.0.1:{args.port}, GPU: {gpu_name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
