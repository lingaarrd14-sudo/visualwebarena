import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
# google-genai is now a required dependency, and creating a Part does not
# require Google Cloud authentication, so we deleted the legacy optional import.
from google.genai import types as genai_types
from PIL import Image


@dataclass
class DetachedPage:
    url: str
    content: str  # html


@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_gemini(img: Image.Image) -> genai_types.Part:
    """Wrap a PNG for google-genai's multimodal generate_content input.

    The legacy Vertex AI SDK used ``VertexImage.from_bytes``; google-genai
    expects the bytes and MIME type in a ``Part`` instead.
    """
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
    return genai_types.Part.from_bytes(data=byte_data, mime_type="image/png")


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None
    center: list[float] | None


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None
    center: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]

Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]
