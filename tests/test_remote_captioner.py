import threading
from http.server import HTTPServer

import pytest
import requests
from PIL import Image

from evaluation_harness import image_utils
from evaluation_harness.image_utils import get_remote_captioning_fn
from scripts.blip2_server import make_handler


def test_remote_captioner_round_trip() -> None:
    calls = []

    def fake_caption(images, prompt, max_new_tokens):
        calls.append((images[0].size, prompt, max_new_tokens))
        return ["a red square"]

    server = HTTPServer(
        ("127.0.0.1", 0),
        make_handler(fake_caption, "Salesforce/blip2-flan-t5-xl", "NVIDIA A5000"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        caption = get_remote_captioning_fn(url, "Salesforce/blip2-flan-t5-xl")
        result = caption([Image.new("RGB", (12, 8), "red")], ["Q: color? A:"], 16)
        assert result == ["a red square"]
        assert calls == [((12, 8), ["Q: color? A:"], 16)]

        with pytest.raises(RuntimeError, match="expected model"):
            get_remote_captioning_fn(url, "another-model")

        response = requests.post(
            f"{url}/caption", json={"model": "wrong", "images": []}, timeout=5
        )
        assert response.status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_configured_captioner_uses_remote_without_loading_local(monkeypatch) -> None:
    marker = object()
    monkeypatch.setenv("VWA_CAPTIONER_URL", "http://127.0.0.1:8765")
    monkeypatch.setattr(
        image_utils, "get_remote_captioning_fn", lambda url, model: (url, model)
    )
    monkeypatch.setattr(
        image_utils, "get_captioning_fn", lambda *args: marker
    )
    assert image_utils.get_configured_captioning_fn(
        "cpu", "float32", "Salesforce/blip2-flan-t5-xl"
    ) == ("http://127.0.0.1:8765", "Salesforce/blip2-flan-t5-xl")
