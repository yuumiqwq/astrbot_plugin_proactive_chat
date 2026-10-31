"""Apply the installed AstrBot context manager to proactive provider requests."""

from __future__ import annotations

from typing import Any

from astrbot.core.agent.context.config import ContextConfig
from astrbot.core.agent.context.manager import ContextManager
from astrbot.core.agent.context.round_utils import split_into_rounds
from astrbot.core.agent.message import Message, TextPart, bind_checkpoint_messages
from astrbot.core.config.agent_runner import resolve_context_compression_config
from astrbot.core.utils.llm_metadata import LLM_METADATAS

from ..domain.proactive_decision import DECISION_INSTRUCTION


def native_qq_enabled(session_config: dict) -> bool:
    settings = session_config.get("context_settings") or {}
    return settings.get("qq_native_context") is True and settings.get(
        "source_mode"
    ) in {"platform_message_history", "hybrid"}


async def make_context_manager(context: Any, session_id: str, provider: Any):
    config = context.get_config(umo=session_id)
    compression = (
        config.get("agent_runner", {}).get("config", {}).get("compression", {})
    )
    options = resolve_context_compression_config(compression)
    max_tokens = provider.provider_config.get("max_context_tokens", 0)
    if max_tokens <= 0:
        metadata = LLM_METADATAS.get(provider.get_model())
        max_tokens = (
            metadata["limit"]["context"]
            if metadata
            else options["fallback_max_context_tokens"]
        )
    compression_provider = None
    if options["context_limit_reached_strategy"] == "llm_compress":
        provider_id = options["llm_compress_provider_id"]
        if provider_id:
            compression_provider = context.get_provider_by_id(provider_id)
        if compression_provider is None:
            compression_provider = provider
    return ContextManager(
        ContextConfig(
            max_context_tokens=max_tokens,
            enforce_max_turns=options["max_context_length"],
            truncate_turns=options["dequeue_context_length"],
            llm_compress_instruction=options["llm_compress_instruction"],
            llm_compress_keep_recent_ratio=options["llm_compress_keep_recent_ratio"],
            llm_compress_provider=compression_provider,
        )
    )


async def manage_request(context: Any, session_id: str, provider: Any, req: Any):
    manager = await make_context_manager(context, session_id, provider)
    history = bind_checkpoint_messages(req.contexts or [])
    current = Message.model_validate(await req.assemble_context())
    # Reserve the final JSON protocol's tokens without giving it to the summary
    # model as a system instruction. It is added to the real system prompt later.
    if isinstance(current.content, str):
        current.content = [TextPart(text=current.content)]
    current.content.append(TextPart(text=DECISION_INSTRUCTION))
    system = (
        Message(role="system", content=req.system_prompt) if req.system_prompt else None
    )
    messages = [system] if system else []
    counter = manager.token_counter
    tokens = counter.count_tokens([*messages, current])
    rounds = split_into_rounds(history)
    for index, round_messages in enumerate(rounds):
        messages.extend(round_messages)
        tokens += counter.count_tokens(round_messages)
        # Replaying a large QQ archive incrementally keeps summary requests near
        # the native compression threshold, as during an ordinary conversation.
        if index < len(rounds) - 1 and manager.compressor.should_compress(
            messages, tokens, manager.config.max_context_tokens
        ):
            processed = await manager.process([*messages, current])
            messages = [item for item in processed if item is not current]
            tokens = counter.count_tokens([*messages, current])
    processed = await manager.process([*messages, current])
    if current not in processed:
        raise ValueError("AstrBot context management removed the current request")
    req.contexts = []
    for message in processed:
        if message is system or message is current:
            continue
        data = message.model_dump()
        if message._no_save:
            data["_no_save"] = True
        req.contexts.append(data)
