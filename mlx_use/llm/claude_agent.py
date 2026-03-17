# pyright: reportUninitializedInstanceVariable=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportOperatorIssue=false, reportArgumentType=false, reportUnusedCallResult=false, reportUnreachable=false
from __future__ import annotations

import asyncio
import importlib
import json
import threading
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import Field

if TYPE_CHECKING:

	class CallbackManagerForLLMRun: ...

	class BaseMessage:
		content: object
		type: str

	class BaseChatModel: ...

	class AIMessage:
		def __init__(
			self,
			*,
			content: str,
			response_metadata: dict[str, object],
			usage_metadata: dict[str, int] | None = None,
		) -> None: ...

	class ChatGeneration:
		def __init__(self, *, message: AIMessage) -> None: ...

	class ChatResult:
		def __init__(self, *, generations: list[ChatGeneration]) -> None: ...

	class SDKTextBlock:
		text: str

	class SDKAssistantMessage:
		content: list[SDKTextBlock]
		usage: dict[str, object] | None

	class SDKResultMessage:
		usage: dict[str, object] | None

	class SDKClaudeAgentOptions:
		def __init__(
			self,
			*,
			model: str,
			system_prompt: str | None,
			max_turns: int,
			cli_path: str | None,
			cwd: str | None,
			permission_mode: str | None,
			**kwargs: object,
		) -> None: ...

	class ClaudeAgentSDKModule(Protocol):
		AssistantMessage: type[SDKAssistantMessage]
		ResultMessage: type[SDKResultMessage]
		TextBlock: type[SDKTextBlock]
		CLINotFoundError: type[Exception]
		ProcessError: type[Exception]
		ClaudeAgentOptions: type[SDKClaudeAgentOptions]

		def query(
			self,
			*,
			prompt: str,
			options: SDKClaudeAgentOptions,
		) -> AsyncIterator[object]: ...
else:
	from langchain_core.callbacks import CallbackManagerForLLMRun
	from langchain_core.language_models.chat_models import BaseChatModel
	from langchain_core.messages import AIMessage, BaseMessage
	from langchain_core.outputs import ChatGeneration, ChatResult

	ClaudeAgentSDKModule = object


class ChatClaudeAgent(BaseChatModel):
	model: str = 'claude-sonnet-4-6'
	cli_path: str | None = None
	cwd: str | None = None
	permission_mode: str | None = None
	system_prompt: str | None = None
	max_turns: int = 1
	kwargs: dict[str, object] = Field(default_factory=dict)

	@property
	def _llm_type(self) -> str:
		return 'claude-agent-sdk'

	@property
	def _identifying_params(self) -> dict[str, object]:
		return {
			'model': self.model,
			'cli_path': self.cli_path,
			'cwd': self.cwd,
			'max_turns': self.max_turns,
		}

	async def _agenerate(
		self,
		messages: list[BaseMessage],
		stop: list[str] | None = None,
		run_manager: CallbackManagerForLLMRun | None = None,
		**kwargs: object,
	) -> ChatResult:
		_ = run_manager
		_ = kwargs
		sdk = self._sdk()

		system_prompt, prompt = self._build_prompt(messages)
		options = sdk.ClaudeAgentOptions(
			model=self.model,
			system_prompt=system_prompt,
			max_turns=self.max_turns,
			cli_path=self.cli_path,
			cwd=self.cwd,
			permission_mode=self.permission_mode,
			**self.kwargs,
		)

		response_text = ''
		usage: dict[str, object] | None = None

		try:
			async for message in sdk.query(prompt=prompt, options=options):
				message_class = message.__class__.__name__
				if message_class == 'AssistantMessage':
					response_text += self._extract_assistant_text(cast('SDKAssistantMessage', message))
					if getattr(message, 'usage', None):
						usage = cast('SDKAssistantMessage', message).usage
				elif message_class == 'ResultMessage' and getattr(message, 'usage', None):
					usage = cast('SDKResultMessage', message).usage
		except Exception as exc:
			raise self._normalize_error(exc, sdk) from exc

		response_text = self._apply_stop_tokens(response_text, stop)
		message = AIMessage(
			content=response_text,
			response_metadata={
				'model': self.model,
				'provider': 'claude-agent-sdk',
			},
			usage_metadata=self._build_usage_metadata(usage),
		)
		return ChatResult(generations=[ChatGeneration(message=message)])

	def _generate(
		self,
		messages: list[BaseMessage],
		stop: list[str] | None = None,
		run_manager: CallbackManagerForLLMRun | None = None,
		**kwargs: object,
	) -> ChatResult:
		return self._run_sync(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))

	def _sdk(self) -> ClaudeAgentSDKModule:
		try:
			module = importlib.import_module('claude_agent_sdk')
			return cast(ClaudeAgentSDKModule, cast(object, module))
		except ImportError as exc:
			raise ImportError('claude-agent-sdk is not installed. Install with `pip install "mlx-use[claude-agent]"`.') from exc

	def _build_prompt(self, messages: list[BaseMessage]) -> tuple[str | None, str]:
		system_parts: list[str] = []
		prompt_parts: list[str] = []

		if self.system_prompt:
			system_parts.append(self.system_prompt)

		for message in messages:
			content = self._stringify_content(message.content)
			tool_calls = getattr(message, 'tool_calls', None)
			if tool_calls:
				tool_call_text = json.dumps(tool_calls, ensure_ascii=True)
				content = f'{content}\nTool calls: {tool_call_text}'.strip()

			message_type = getattr(message, 'type', message.__class__.__name__.lower())
			if message_type == 'system':
				if content:
					system_parts.append(content)
				continue

			label = self._message_label(message_type)
			prompt_parts.append(f'{label}: {content}'.rstrip())

		prompt_parts.append('Assistant:')
		system_prompt = '\n\n'.join(part for part in system_parts if part) or None
		prompt = '\n\n'.join(part for part in prompt_parts if part)
		return system_prompt, prompt

	def _message_label(self, message_type: str) -> str:
		mapping = {
			'human': 'Human',
			'ai': 'Assistant',
			'tool': 'Tool',
			'function': 'Tool',
		}
		return mapping.get(message_type, message_type.capitalize())

	def _stringify_content(self, content: object) -> str:
		if isinstance(content, str):
			return content
		if isinstance(content, list):
			parts: list[str] = []
			for item in content:
				if isinstance(item, str):
					parts.append(item)
				elif isinstance(item, dict):
					text = item.get('text')
					if isinstance(text, str):
						parts.append(text)
					else:
						parts.append(json.dumps(item, ensure_ascii=True))
				else:
					parts.append(str(item))
			return '\n'.join(part for part in parts if part)
		if content is None:
			return ''
		return str(content)

	def _extract_assistant_text(self, message: SDKAssistantMessage) -> str:
		return ''.join(block.text for block in message.content if block.__class__.__name__ == 'TextBlock')

	def _build_usage_metadata(self, usage: dict[str, object] | None) -> dict[str, int] | None:
		if not usage:
			return None

		input_tokens = self._coerce_int(usage.get('input_tokens') or usage.get('inputTokens') or 0)
		output_tokens = self._coerce_int(usage.get('output_tokens') or usage.get('outputTokens') or 0)
		total_tokens = usage.get('total_tokens') or usage.get('totalTokens') or (input_tokens + output_tokens)
		return {
			'input_tokens': input_tokens,
			'output_tokens': output_tokens,
			'total_tokens': self._coerce_int(total_tokens),
		}

	def _coerce_int(self, value: object) -> int:
		if isinstance(value, bool):
			return int(value)
		if isinstance(value, int):
			return value
		if isinstance(value, float):
			return int(value)
		if isinstance(value, str):
			return int(value)
		return 0

	def _apply_stop_tokens(self, text: str, stop: list[str] | None) -> str:
		if not stop:
			return text

		stop_positions = [text.find(token) for token in stop if token]
		stop_positions = [position for position in stop_positions if position >= 0]
		if not stop_positions:
			return text
		return text[: min(stop_positions)]

	def _normalize_error(self, exc: Exception, sdk: ClaudeAgentSDKModule) -> Exception:
		if isinstance(exc, sdk.CLINotFoundError):
			return RuntimeError(
				'Claude CLI was not found for claude-agent-sdk. Install the optional dependency or set `cli_path` to a valid Claude binary.'
			)
		if isinstance(exc, sdk.ProcessError):
			return RuntimeError(f'Claude Agent SDK request failed: {exc}')
		return exc

	def _run_sync(self, coroutine: Coroutine[object, object, ChatResult]) -> ChatResult:
		try:
			asyncio.get_running_loop()
		except RuntimeError:
			return asyncio.run(coroutine)

		result: ChatResult | None = None
		error: Exception | None = None

		def runner() -> None:
			nonlocal result, error
			loop = asyncio.new_event_loop()
			try:
				asyncio.set_event_loop(loop)
				result = loop.run_until_complete(coroutine)
			except Exception as exc:
				error = exc
			finally:
				asyncio.set_event_loop(None)
				loop.close()

		thread = threading.Thread(target=runner, daemon=True)
		thread.start()
		thread.join()

		if error is not None:
			raise error
		if result is None:
			raise RuntimeError('Claude Agent SDK request returned no result.')
		return result
