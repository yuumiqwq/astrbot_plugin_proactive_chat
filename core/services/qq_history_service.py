"""Read QQ private history as chronological user/assistant messages."""

from __future__ import annotations

import asyncio
from typing import Any


def qq_client(plugin: Any, session_id: str) -> Any | None:
    parts = session_id.split(":", 2)
    if len(parts) != 3 or parts[1] != "FriendMessage" or not parts[2].isdigit():
        return None
    platform = plugin.context.get_platform_inst(parts[0])
    if platform is None or platform.meta().name != "aiocqhttp":
        return None
    return platform.get_client()


def message_key(message: dict) -> str:
    return str(message.get("message_id") or message.get("real_id") or "")


def message_order(message: dict) -> tuple[int, int]:
    return int(message.get("time") or 0), int(message.get("real_seq") or 0)


async def read_qq_history(client: Any, user_id: int) -> list[dict]:
    """Page backwards until QQ has no older records; page size is not a total cap.

    NapCat accepts its returned message_id as the message_seq pagination cursor.
    Pages may overlap or contain fewer records than requested before history ends.
    """
    records: dict[str, dict] = {}
    cursor = ""
    cursors: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "user_id": user_id,
            "count": 100,
            "reverse_order": True,
            "disable_get_url": True,
            "parse_mult_msg": False,
        }
        if cursor:
            params["message_seq"] = cursor
        try:
            response = await asyncio.wait_for(
                client.call_action("get_friend_msg_history", **params), timeout=30
            )
        except Exception as error:
            # NapCat reports an empty history page as an action error.
            if cursor and "消息" in str(error) and "不存在" in str(error):
                break
            raise
        messages = response.get("messages")
        if not isinstance(messages, list):
            raise ValueError("QQ history response has no messages list")
        if not messages:
            break
        page = sorted(messages, key=message_order)
        for message in page:
            key = message_key(message)
            if not key:
                raise ValueError("QQ history record has no pagination identity")
            records[key] = message
        # QQ's first-view response can be sparse across offline periods. Start
        # the contiguous backward scan at its newest record, not its oldest.
        next_cursor = message_key(page[0] if cursor else page[-1])
        if next_cursor == cursor or next_cursor in cursors:
            break
        cursors.add(next_cursor)
        cursor = next_cursor
    return sorted(records.values(), key=message_order)


def history_as_dialogue(
    records: list[dict], context_service: Any, include_bot_messages: bool
) -> list[dict]:
    """Keep QQ bubble boundaries and text whitespace, including example style."""
    messages = []
    for record in records:
        sender = record.get("sender") or {}
        sender_id = str(sender.get("user_id") or record.get("user_id") or "")
        is_bot = sender_id == str(record.get("self_id") or "")
        if is_bot and not include_bot_messages:
            continue
        content = record.get("message") or record.get("raw_message") or ""
        if isinstance(content, list):
            content = [
                {**(part.get("data") or {}), "type": part.get("type", "")}
                for part in content
                if isinstance(part, dict)
            ]
        text = context_service.extract_platform_message_text(content)
        if text:
            messages.append(
                {"role": "assistant" if is_bot else "user", "content": text}
            )
    return messages
