"""The reasoning + action output contract and its browser Action adapter."""

import json
import re
from collections.abc import Callable
from typing import Any

from browser_env import actions

SCHEMA_VERSION = "browser_action_reasoning_v1"
_STRING = {"type": "string"}
_ARGUMENTS = {
    "click": {"element_id": _STRING},
    "hover": {"element_id": _STRING},
    "type": {
        "element_id": _STRING,
        "text": _STRING,
        "press_enter_after": {"type": "boolean"},
    },
    "press": {"key_comb": _STRING},
    "scroll": {"direction": {"type": "string", "enum": ["up", "down"]}},
    "goto": {"url": _STRING},
    "tab_focus": {"tab_index": {"type": "integer", "minimum": 0}},
    "new_tab": {},
    "close_tab": {},
    "go_back": {},
    "go_forward": {},
    "stop": {"answer": _STRING},
}


def _closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


ACTION_RESPONSE_SCHEMA = _closed_object({
    "reasoning": _STRING,
    "action": {"anyOf": [
        _closed_object({"type": {"type": "string", "enum": [name]}, **fields})
        for name, fields in _ARGUMENTS.items()
    ]},
})

_FACTORIES = {
    "click": actions.create_click_action,
    "hover": actions.create_hover_action,
    "type": actions.create_type_action,
    "press": actions.create_key_press_action,
    "scroll": actions.create_scroll_action,
    "goto": actions.create_goto_url_action,
    "tab_focus": actions.create_page_focus_action,
    "new_tab": actions.create_new_tab_action,
    "close_tab": actions.create_page_close_action,
    "go_back": actions.create_go_back_action,
    "go_forward": actions.create_go_forward_action,
    "stop": actions.create_stop_action,
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _validate(value: Any) -> dict[str, Any]:
    """Validate this fixed contract using the same field definitions as the schema."""
    if type(value) is not dict or set(value) != {"reasoning", "action"}:
        raise ValueError("Expected exactly reasoning and action")
    if type(value["reasoning"]) is not str:
        raise ValueError("reasoning must be a string")
    action = value["action"]
    if type(action) is not dict or type(action.get("type")) is not str:
        raise ValueError("action must be an object with a string type")
    if action["type"] not in _ARGUMENTS:
        raise ValueError(f"Unknown action type: {action['type']}")
    fields = _ARGUMENTS[action["type"]]
    if set(action) != {"type", *fields}:
        raise ValueError("Missing or extra action fields")
    for name, schema in fields.items():
        field = action[name]
        expected_type = {"string": str, "integer": int, "boolean": bool}[schema["type"]]
        if type(field) is not expected_type:
            raise ValueError(f"Wrong type for {name}")
        if "enum" in schema and field not in schema["enum"]:
            raise ValueError(f"Invalid value for {name}")
        if "minimum" in schema and field < schema["minimum"]:
            raise ValueError(f"Below minimum for {name}")
    if "element_id" in action and not re.fullmatch(r"[0-9]+", action["element_id"]):
        raise ValueError("element_id must contain only digits")
    return action


def parse_action_response(
    response: str,
    map_url_to_local: Callable[[str], str],
) -> actions.Action:
    """Decode the entire response and call existing factories without a DSL round trip."""
    try:
        value = json.loads(
            response, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        action = _validate(value)
    except ValueError as exc:
        raise actions.ActionParsingError(f"Invalid structured action: {exc}") from exc

    kind = action["type"]
    arguments = {
        name: map_url_to_local(value) if isinstance(value, str) else value
        for name, value in action.items() if name != "type"
    }
    if kind == "type":
        if arguments.pop("press_enter_after"):
            arguments["text"] += "\n"
    elif kind == "tab_focus":
        arguments["page_number"] = arguments.pop("tab_index")
    result = _FACTORIES[kind](**arguments)
    result["raw_prediction"] = response
    return result
