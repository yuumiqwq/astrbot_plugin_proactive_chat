"""Verify QQ pagination and the real AstrBot length manager without network calls."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message

from astrbot_plugin_proactive_chat.core.domain.proactive_decision import (
    DECISION_INSTRUCTION,
)
from astrbot_plugin_proactive_chat.core.services.context_service import ContextService
from astrbot_plugin_proactive_chat.core.services.llm_service import LlmService
from astrbot_plugin_proactive_chat.core.services.native_context_service import (
    make_context_manager,
    manage_request,
)
from astrbot_plugin_proactive_chat.core.services.qq_history_service import (
    history_as_dialogue,
    read_qq_history,
)


def record(seq, text=None, self_sent=False):
    return {
        "message_id": seq * 7,
        "real_seq": str(seq),
        "time": seq,
        "self_id": 200,
        "sender": {"user_id": 200 if self_sent else 100},
        "message": [{"type": "text", "data": {"text": text or f"message {seq}"}}],
    }


class QQHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_sparse_first_view_and_overlapping_short_pages(self):
        all_messages = [record(i) for i in range(1, 258)]

        async def api(action, **params):
            self.assertEqual(action, "get_friend_msg_history")
            self.assertTrue(params["reverse_order"])
            if "message_seq" not in params:
                return {
                    "messages": [all_messages[0], all_messages[156], all_messages[-1]]
                }
            seq = int(params["message_seq"]) // 7
            # Short pages do not mean the archive has ended.
            return {"messages": all_messages[max(0, seq - 37) : seq]}

        client = SimpleNamespace(call_action=AsyncMock(side_effect=api))
        result = await read_qq_history(client, 100)
        self.assertEqual(result, all_messages)
        self.assertEqual(
            client.call_action.call_args_list[1].kwargs["message_seq"], str(257 * 7)
        )

    async def test_error_on_end_page_and_network_failure(self):
        client = SimpleNamespace(
            call_action=AsyncMock(
                side_effect=[{"messages": [record(1)]}, RuntimeError("消息7不存在")]
            )
        )
        self.assertEqual(await read_qq_history(client, 100), [record(1)])
        client.call_action = AsyncMock(side_effect=TimeoutError("offline"))
        with self.assertRaises(TimeoutError):
            await read_qq_history(client, 100)

    async def test_native_source_ignores_old_caps_and_keeps_paragraphs(self):
        records = [
            record(i, "第一段\n\n第二段" + "文" * 100, i % 2 == 0) for i in range(1, 80)
        ]
        client = SimpleNamespace(
            call_action=AsyncMock(
                side_effect=[
                    {"messages": records},
                    {"messages": records},
                    {"messages": [records[0]]},
                ]
            )
        )
        platform = SimpleNamespace(
            meta=lambda: SimpleNamespace(name="aiocqhttp"), get_client=lambda: client
        )
        context = SimpleNamespace(get_platform_inst=lambda _: platform)
        service = ContextService(SimpleNamespace(context=context))
        settings = {
            "source_mode": "platform_message_history",
            "qq_native_context": True,
            "include_bot_messages": True,
            "platform_history_count": 20,
            "platform_context_max_chars": 4000,
        }
        history, platform_text = await service.build_effective_history_context(
            "test:FriendMessage:100", [{"role": "user", "content": "stale"}], settings
        )
        self.assertEqual(len(history), 79)
        self.assertGreater(sum(len(m["content"]) for m in history), 4000)
        self.assertEqual(history[1]["role"], "assistant")
        self.assertIn("\n\n", history[1]["content"])
        self.assertEqual(platform_text, "")

    async def test_empty_qq_history_does_not_restore_deleted_astrbot_history(self):
        client = SimpleNamespace(call_action=AsyncMock(return_value={"messages": []}))
        platform = SimpleNamespace(
            meta=lambda: SimpleNamespace(name="aiocqhttp"), get_client=lambda: client
        )
        service = ContextService(
            SimpleNamespace(
                context=SimpleNamespace(get_platform_inst=lambda _: platform)
            )
        )
        history, _ = await service.build_effective_history_context(
            "test:FriendMessage:100",
            [{"role": "user", "content": "stale"}],
            {
                "source_mode": "platform_message_history",
                "qq_native_context": True,
                "include_bot_messages": True,
            },
        )
        self.assertEqual(history, [])

    def test_empty_bubbles_and_bot_filter(self):
        empty = record(1)
        empty["message"] = []
        result = history_as_dialogue(
            [empty, record(2, self_sent=True), record(3)], ContextService(None), False
        )
        self.assertEqual(result, [{"role": "user", "content": "message 3"}])


class NativeContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = {
            "max_turns": -1,
            "trim_turns": 1,
            "overflow_strategy": "llm_compress",
            "instruction": "Keep the conversation facts.",
            "keep_recent_ratio": 0.15,
            "provider_id": "",
            "fallback_max_tokens": 128000,
        }
        self.context = SimpleNamespace(
            get_config=lambda **_: {
                "agent_runner": {"config": {"compression": self.settings}}
            },
            get_provider_by_id=Mock(return_value=None),
        )
        self.provider = SimpleNamespace(
            provider_config={"max_context_tokens": 128000},
            get_model=lambda: "test-model",
            text_chat=AsyncMock(
                return_value=LLMResponse(
                    role="assistant", completion_text="A short summary."
                )
            ),
        )

    def request(self, turns=30, size=100):
        return ProviderRequest(
            prompt="Decide whether to contact the user.",
            system_prompt="Keep the requested personality.",
            contexts=[
                {"role": role, "content": f"round {i} {role}: " + "文" * size}
                for i in range(turns)
                for role in ("user", "assistant")
            ],
        )

    async def test_under_native_budget_is_not_cut_to_20_or_4000(self):
        req = self.request()
        original = list(req.contexts)
        await manage_request(self.context, "test", self.provider, req)
        self.assertEqual(req.contexts, original)
        self.provider.text_chat.assert_not_awaited()

    async def test_resolves_session_settings_and_model_capacity(self):
        self.provider.provider_config = {}
        self.settings.update(
            max_turns=9,
            trim_turns=2,
            provider_id="summary-model",
            keep_recent_ratio=0.2,
        )
        summary_provider = SimpleNamespace()
        self.context.get_provider_by_id.return_value = summary_provider
        with patch.dict(
            "astrbot_plugin_proactive_chat.core.services.native_context_service.LLM_METADATAS",
            {"test-model": {"limit": {"context": 1000000}}},
            clear=True,
        ):
            manager = await make_context_manager(self.context, "test", self.provider)
        self.assertEqual(manager.config.max_context_tokens, 1000000)
        self.assertEqual(manager.config.enforce_max_turns, 9)
        self.assertEqual(manager.config.truncate_turns, 2)
        self.assertEqual(manager.config.llm_compress_keep_recent_ratio, 0.2)
        self.assertIs(manager.config.llm_compress_provider, summary_provider)
        with patch.dict(
            "astrbot_plugin_proactive_chat.core.services.native_context_service.LLM_METADATAS",
            {},
            clear=True,
        ):
            manager = await make_context_manager(self.context, "test", self.provider)
        self.assertEqual(manager.config.max_context_tokens, 128000)

    async def test_native_turn_limit(self):
        self.settings["max_turns"] = 3
        req = self.request(turns=20)
        latest = req.contexts[-1]
        await manage_request(self.context, "test", self.provider, req)
        self.assertEqual(req.contexts[-1], latest)
        # The active proactive request also occupies one native user round.
        self.assertLessEqual(sum(m["role"] == "user" for m in req.contexts), 2)

    async def test_native_summary_for_large_archive_preserves_latest(self):
        self.provider.provider_config["max_context_tokens"] = 10000
        req = self.request(turns=80, size=180)
        latest = req.contexts[-2:]
        await manage_request(self.context, "test", self.provider, req)
        self.assertGreater(self.provider.text_chat.await_count, 1)
        self.assertEqual(req.contexts[-2:], latest)
        self.assertTrue(
            any(
                "previous history conversation summary" in m["content"]
                for m in req.contexts
            )
        )
        counter = EstimateTokenCounter()
        for call in self.provider.text_chat.call_args_list:
            messages = [
                m if isinstance(m, Message) else Message.model_validate(m)
                for m in call.kwargs["contexts"]
            ]
            self.assertLess(counter.count_tokens(messages), 10000)
            self.assertNotIn(DECISION_INSTRUCTION, str(call.kwargs["contexts"]))

    async def test_summary_failure_uses_native_truncation(self):
        self.provider.provider_config["max_context_tokens"] = 3000
        self.provider.text_chat.side_effect = RuntimeError("unavailable")
        req = self.request(turns=20, size=250)
        latest = req.contexts[-2:]
        await manage_request(self.context, "test", self.provider, req)
        self.assertEqual(req.contexts[-2:], latest)
        self.assertLess(len(req.contexts), 40)

    async def test_pipeline_manages_after_hooks_and_keeps_silence_protocol(self):
        config = {
            "proactive_prompt": "Decide whether to send.",
            "context_settings": {
                "source_mode": "platform_message_history",
                "qq_native_context": True,
            },
        }
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant", completion_text='{"send":false,"message":""}'
        )
        plugin = SimpleNamespace(
            context=self.context, timezone=ZoneInfo("Asia/Shanghai"), telemetry=None
        )
        plugin.context_service = ContextService(plugin)
        service = LlmService(plugin)
        service.resolve_chat_provider = AsyncMock(return_value=self.provider)

        async def hook(event, req):
            req.system_prompt = "Memory and personality from hooks."
            return False

        service.dispatch_llm_request_hooks = AsyncMock(side_effect=hook)
        service.dispatch_llm_response_hooks = AsyncMock()
        history = self.request().contexts
        result, _ = await service.generate_llm_response(
            "test", config, history, "persona", 0
        )
        self.assertIsNone(result)
        call = self.provider.text_chat.call_args.kwargs
        self.assertEqual(len(call["contexts"]), 60)
        self.assertIn("Memory and personality from hooks.", call["system_prompt"])
        self.assertIn(DECISION_INSTRUCTION, call["system_prompt"])
        service.dispatch_llm_response_hooks.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
