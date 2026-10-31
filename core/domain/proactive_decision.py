"""主动消息的 JSON 输出协议与发送决定。"""

from __future__ import annotations

import json
from dataclasses import dataclass


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "send": {"type": "boolean"},
        "message": {"type": "string"},
    },
    "required": ["send", "message"],
    "additionalProperties": False,
    "oneOf": [
        {"properties": {"send": {"const": False}, "message": {"const": ""}}},
        {"properties": {"send": {"const": True}, "message": {"pattern": r"\S"}}},
    ],
}

DECISION_INSTRUCTION = (
    "[本轮主动消息输出协议]\n"
    "先根据上下文判断是否适合主动联系。可以选择保持沉默。\n"
    "本轮只输出一个符合以下 JSON Schema 的 JSON 对象；"
    "人格、预设对话和文风要求作用于 message 中的聊天正文。\n"
    + json.dumps(DECISION_SCHEMA, ensure_ascii=False)
    + '\n决定不发送时，输出 {"send":false,"message":""}。'
    '\n决定发送时，输出 {"send":true,"message":"实际要发送的正文"}。'
    "\n正文中的段落用 JSON 字符串内的 \\n\\n 表示。"
    "\n不要添加代码围栏或 JSON 之外的解释；此前提示中的空输出要求"
    "统一用 send 为 false 的对象表示。"
)


@dataclass(frozen=True)
class ProactiveDecision:
    send: bool
    message: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def parse_decision(text: str) -> ProactiveDecision:
    """校验完整 JSON；仅 send=true 且正文非空的结果允许发送。"""
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid_json") from error
    if not isinstance(value, dict) or set(value) != {"send", "message"}:
        raise ValueError("invalid_fields")
    if type(value["send"]) is not bool or not isinstance(value["message"], str):
        raise ValueError("invalid_field_types")
    message = value["message"]
    if not value["send"]:
        if message != "":
            raise ValueError("nonempty_silent_message")
        return ProactiveDecision(send=False, message="")
    if not message.strip():
        raise ValueError("empty_message")
    return ProactiveDecision(send=True, message=message.strip())
