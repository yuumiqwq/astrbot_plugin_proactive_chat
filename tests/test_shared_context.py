"""Proactive requests delegate history management to AstrBot."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from astrbot.api.provider import LLMResponse
from astrbot_plugin_proactive_chat.core.domain.proactive_decision import (
    DECISION_INSTRUCTION,
)
from astrbot_plugin_proactive_chat.core.services.context_service import ContextService
from astrbot_plugin_proactive_chat.core.services.llm_service import LlmService


class SharedContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_core_prepares_request_after_hooks(self):
        provider = SimpleNamespace(
            text_chat=AsyncMock(
                return_value=LLMResponse(
                    role="assistant", completion_text='{"send":false,"message":""}'
                )
            )
        )

        async def prepare(session_id, selected, request, response_instruction):
            self.assertIs(selected, provider)
            self.assertEqual(request.system_prompt, "Persona and recalled memory")
            self.assertEqual(response_instruction, DECISION_INSTRUCTION)
            request.contexts = [{"role": "user", "content": "Shared summary"}]
            request.system_prompt += "\n" + response_instruction

        context = SimpleNamespace(
            prepare_session_request=AsyncMock(side_effect=prepare)
        )
        plugin = SimpleNamespace(
            context=context, timezone=ZoneInfo("Asia/Shanghai"), telemetry=None
        )
        plugin.context_service = ContextService(plugin)
        service = LlmService(plugin)
        service.resolve_chat_provider = AsyncMock(return_value=provider)

        async def hooks(event, request):
            request.system_prompt = "Persona and recalled memory"
            return False

        service.dispatch_llm_request_hooks = AsyncMock(side_effect=hooks)
        service.dispatch_llm_response_hooks = AsyncMock()
        result, _ = await service.generate_llm_response(
            "test:FriendMessage:100",
            {"proactive_prompt": "Decide whether to send"},
            [{"role": "user", "content": "Unmanaged history"}],
            "Persona",
            0,
        )
        self.assertIsNone(result)
        context.prepare_session_request.assert_awaited_once()
        self.assertEqual(
            provider.text_chat.call_args.kwargs["contexts"],
            [{"role": "user", "content": "Shared summary"}],
        )
        service.dispatch_llm_response_hooks.assert_not_awaited()

    async def test_core_failure_does_not_bypass_management(self):
        provider = SimpleNamespace(text_chat=AsyncMock())
        context = SimpleNamespace(
            prepare_session_request=AsyncMock(side_effect=RuntimeError("unavailable"))
        )
        plugin = SimpleNamespace(
            context=context, timezone=ZoneInfo("Asia/Shanghai"), telemetry=None
        )
        plugin.context_service = ContextService(plugin)
        service = LlmService(plugin)
        service.resolve_chat_provider = AsyncMock(return_value=provider)
        service.dispatch_llm_request_hooks = AsyncMock(return_value=False)
        result, _ = await service.generate_llm_response("session", {}, [], "Persona", 0)
        self.assertIsNone(result)
        provider.text_chat.assert_not_awaited()
