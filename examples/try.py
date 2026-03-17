# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportCallIssue=false
import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from mlx_use import Agent
from mlx_use.llm import ChatClaudeAgent
from pydantic import SecretStr
from mlx_use.controller.service import Controller


def set_llm(llm_provider: str | None = None):
	if not llm_provider:
		raise ValueError('No llm provider was set')

	openai_api_key = os.getenv('OPENAI_API_KEY')
	if llm_provider == 'OAI' and openai_api_key:
		return ChatOpenAI(model='gpt-4', api_key=SecretStr(openai_api_key))

	gemini_api_key = os.getenv('GEMINI_API_KEY')
	if llm_provider == 'google' and gemini_api_key:
		return ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(gemini_api_key))

	anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
	if llm_provider == 'anthropic' and anthropic_api_key:
		return ChatAnthropic(model='claude-3-sonnet-20240229', api_key=SecretStr(anthropic_api_key))

	if llm_provider == 'claude-agent' and os.getenv('CLAUDE_AGENT') == '1':
		return ChatClaudeAgent(
			model=os.getenv('CLAUDE_AGENT_MODEL', 'claude-sonnet-4-6'),
			cli_path=os.getenv('CLAUDE_AGENT_CLI_PATH'),
		)

	return None


# Try to set LLM based on available API keys
llm = None
if os.getenv('CLAUDE_AGENT') == '1':
	llm = set_llm('claude-agent')
elif os.getenv('GEMINI_API_KEY'):
	llm = set_llm('google')
elif os.getenv('OPENAI_API_KEY'):
	llm = set_llm('OAI')
elif os.getenv('ANTHROPIC_API_KEY'):
	llm = set_llm('anthropic')

if not llm:
	raise ValueError(
		'No LLM provider configured. Set `CLAUDE_AGENT=1` for Claude CLI auth, or provide one of GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in your .env file.'
	)

controller = Controller()


async def main():

	agent_greeting = Agent(
		task='Say "Hi there $whoami,  What can I do for you today?"',
		llm=llm,
		controller=controller,
		use_vision=False,
		max_actions_per_step=1,
		max_failures=5,
	)

	_ = await agent_greeting.run(max_steps=25)
	task = input('Enter the task: ')

	agent_task = Agent(task=task, llm=llm, controller=controller, use_vision=False, max_actions_per_step=4, max_failures=5)

	_ = await agent_task.run(max_steps=25)


asyncio.run(main())
