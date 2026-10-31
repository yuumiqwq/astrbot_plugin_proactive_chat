"""Exercise structured decisions with AstrBot response objects and the real pipeline.

Run in an AstrBot environment with the plugins directory on PYTHONPATH:
python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

from astrbot.api.message_components import Plain
from astrbot.api.provider import LLMResponse
from astrbot.core.message.message_event_result import MessageChain

from astrbot_plugin_proactive_chat.core.domain.proactive_decision import (
    DECISION_INSTRUCTION,
    parse_decision,
)
from astrbot_plugin_proactive_chat.core.domain.request_context import ProactiveContext
from astrbot_plugin_proactive_chat.core.features.deliver.split_delivery import (
    DeliverStage,
)
from astrbot_plugin_proactive_chat.core.features.prompt.single_prompt import (
    DeliberateStage,
)
from astrbot_plugin_proactive_chat.core.features.record.history_writer import (
    RecordStage,
)
from astrbot_plugin_proactive_chat.core.pipeline.hook_registry import HookRegistry
from astrbot_plugin_proactive_chat.core.pipeline.pipeline_runner import PipelineRunner
from astrbot_plugin_proactive_chat.core.services.llm_service import LlmService
from astrbot_plugin_proactive_chat.core.services.sender_service import SenderService


class DecisionParserTests(unittest.TestCase):
    def test_silence(self):
        decision = parse_decision('{"send":false,"message":""}')
        self.assertFalse(decision.send)
        self.assertEqual(decision.message, "")

    def test_body_and_paragraphs(self):
        decision = parse_decision(
            json.dumps({"send": True, "message": "  在吗？\n\n刚看到“这个”。  "})
        )
        self.assertTrue(decision.send)
        self.assertEqual(decision.message, "在吗？\n\n刚看到“这个”。")

    def test_invalid_outputs_are_rejected(self):
        invalid = [
            "",
            " \n ",
            '""',
            "不发送",
            "null",
            "[]",
            "{}",
            '{"send":"false","message":""}',
            '{"send":1,"message":"hello"}',
            '{"send":true,"message":null}',
            '{"send":true,"message":[]}',
            '{"send":true,"message":""}',
            '{"send":true,"message":" \\n\\t "}',
            '{"send":false,"message":"do not send this"}',
            '{"send":false,"message":" "}',
            '{"send":true}',
            '{"send":false,"message":"","reason":"busy"}',
            '{"send":false,"send":true,"message":"hello"}',
            '{"send":true,"message":"hello"} trailing text',
            '```json\n{"send":false,"message":""}\n```',
            '{"send":true,"message":"truncated',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_decision(text)


class DecisionPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "_session_type": "friend",
            "proactive_prompt": "请自然地决定是否联系。",
            "tts_settings": {"enable_tts": False},
            "segmented_reply_settings": {
                "enable": True,
                "words_count_threshold": 1000000,
                "regex": r"\S[\s\S]*?(?=(?:\r?\n[^\S\r\n]*){2,}|\Z)",
            },
        }
        self.provider = SimpleNamespace(text_chat=AsyncMock())
        self.plugin = SimpleNamespace(
            timezone=ZoneInfo("Asia/Shanghai"),
            telemetry=None,
            context_service=SimpleNamespace(
                sanitize_history_content=lambda history: list(history),
                make_temp_text_part=lambda text: {"type": "text", "text": text},
            ),
            config_service=SimpleNamespace(
                get_session_config=lambda session_id: self.config
            ),
            session_service=SimpleNamespace(
                get_session_log_str=lambda *args: "test-session"
            ),
        )
        self.service = LlmService(self.plugin)
        self.service.resolve_chat_provider = AsyncMock(return_value=self.provider)
        self.service.dispatch_llm_request_hooks = AsyncMock(return_value=False)
        self.service.dispatch_llm_response_hooks = AsyncMock(return_value=False)
        self.plugin.llm_service = self.service
        self.sender = SenderService(self.plugin)
        self.sender.build_proactive_event = Mock(return_value=None)
        self.sender.run_decorating_hooks = AsyncMock(
            side_effect=lambda event, parts: (parts, True)
        )
        self.sender.send_chain = AsyncMock(return_value=True)
        self.sender.calc_interval = AsyncMock(return_value=0)
        self.plugin.sender_service = self.sender
        self.orchestrator = SimpleNamespace(
            plugin=self.plugin, finalize_and_reschedule=AsyncMock()
        )
        registry = HookRegistry()
        for stage in (DeliberateStage, DeliverStage, RecordStage):
            registry.register(stage(self.orchestrator))
        self.runner = PipelineRunner(registry)
        self.reschedule = AsyncMock()
        self.runner.set_reschedule_handler(self.reschedule)
        self.ctx = ProactiveContext(
            session_id="test:FriendMessage:1",
            conversation_id="test-conversation",
            unanswered_count=2,
            extra={
                "history": [{"role": "user", "content": "history"}],
                "system_prompt": "人格与预设文风。",
            },
        )

    async def run_output(self, text, *, chain=False):
        self.provider.text_chat.return_value = LLMResponse(
            role="assistant",
            completion_text=None if chain else text,
            result_chain=MessageChain([Plain(text)]) if chain else None,
        )
        return await self.runner.run(self.ctx)

    async def assert_no_delivery(self, text, *, chain=False):
        await self.run_output(text, chain=chain)
        self.sender.send_chain.assert_not_awaited()
        self.sender.run_decorating_hooks.assert_not_awaited()
        self.orchestrator.finalize_and_reschedule.assert_not_awaited()
        self.service.dispatch_llm_response_hooks.assert_not_awaited()
        self.reschedule.assert_awaited_once()
        self.assertEqual(self.ctx.unanswered_count, 2)

    async def test_silent_decision_never_reaches_sender_or_memory_hooks(self):
        await self.assert_no_delivery('{"send":false,"message":""}')

    async def test_silent_result_chain_cannot_bypass_decision(self):
        await self.assert_no_delivery('{"send":false,"message":""}', chain=True)

    async def test_empty_completion_does_not_send_empty_message(self):
        await self.assert_no_delivery("")

    async def test_empty_chain_does_not_send_empty_message(self):
        await self.assert_no_delivery(" \n ", chain=True)

    async def test_invalid_json_does_not_get_sent_as_text(self):
        await self.assert_no_delivery("普通文字但缺少发送决定")

    async def test_send_true_with_empty_message_is_skipped(self):
        await self.assert_no_delivery('{"send":true,"message":""}')

    async def test_only_body_is_delivered_and_recorded(self):
        await self.run_output('{"send":true,"message":"在吗？"}', chain=True)
        self.sender.send_chain.assert_awaited_once()
        parts = self.sender.send_chain.call_args.args[2]
        self.assertEqual([part.text for part in parts], ["在吗？"])
        hook_response = self.service.dispatch_llm_response_hooks.call_args.args[1]
        self.assertEqual(hook_response.completion_text, "在吗？")
        self.orchestrator.finalize_and_reschedule.assert_awaited_once_with(
            self.ctx.session_id,
            self.ctx.conversation_id,
            self.config["proactive_prompt"],
            "在吗？",
            2,
        )
        self.reschedule.assert_not_awaited()

    async def test_paragraphs_become_separate_messages(self):
        await self.run_output(
            json.dumps({"send": True, "message": "在的在的\n\n都两点多了。还没睡啊"})
        )
        self.assertEqual(self.sender.send_chain.await_count, 2)
        messages = [
            call.args[2][0].text for call in self.sender.send_chain.call_args_list
        ]
        self.assertEqual(messages, ["在的在的", "都两点多了。还没睡啊"])

    async def test_contract_added_after_request_hooks_and_not_saved_in_prompt(self):
        def update_system(event, request):
            request.system_prompt = "钩子处理后的人格与记忆。"
            return False

        self.service.dispatch_llm_request_hooks.side_effect = update_system
        await self.run_output('{"send":true,"message":"hello"}')
        request = self.provider.text_chat.call_args.kwargs
        self.assertTrue(request["system_prompt"].startswith("钩子处理后的人格与记忆。"))
        self.assertTrue(request["system_prompt"].endswith(DECISION_INSTRUCTION))
        self.assertEqual(request["prompt"], self.config["proactive_prompt"])
        self.assertEqual(self.ctx.extra["final_user_prompt"], request["prompt"])
        self.assertEqual(request["contexts"], [{"role": "user", "content": "history"}])

    async def test_response_hook_clearing_text_still_sends_nothing(self):
        def clear_response(event, response):
            response.result_chain = MessageChain([Plain(" \n ")])
            return False

        self.service.dispatch_llm_response_hooks.side_effect = clear_response
        await self.run_output('{"send":true,"message":"hello"}')
        self.sender.send_chain.assert_not_awaited()
        self.orchestrator.finalize_and_reschedule.assert_not_awaited()
        self.reschedule.assert_awaited_once()

    async def test_provider_exception_reschedules_without_sending(self):
        self.provider.text_chat.side_effect = RuntimeError("offline test error")
        await self.assert_no_delivery('{"send":true,"message":"hello"}')


if __name__ == "__main__":
    unittest.main()
