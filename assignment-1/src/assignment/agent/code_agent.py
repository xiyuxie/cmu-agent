"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        # self.tools is the list of tools the model may use.
        # Agent.query_language_model() sends it to the model with every
        # request. The model reads the list and replies either with tool calls
        # (the tool it picked and its arguments) or with text only. The tool
        # calls are then run by execute_tool_calls() below.
        # Keep anything the base class already added (invoke_skill).
        self.tools = [EXECUTE_TOOL, SEND_MESSAGE_TOOL, *self.tools]

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations = []
        for call in tool_calls:
            name = call["function"]["name"]

            # 1. The arguments arrive as a JSON string. Turn them into a dict.
            try:
                arguments = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = None

            # 2. Decide what to do, and put the result text in `content`.
            if arguments is None:
                content = (
                    "<error>The arguments were not valid JSON. "
                    "Nothing was run. Try again.</error>"
                )
            elif name == "execute":
                command = arguments.pop("command")
                options = {
                    key: value for key, value in arguments.items() if value is not None
                }
                result = self.env.execute(command, **options)
                content = format_tool_output(result)
            elif name == "send_message":
                self.finished = True
                content = "Message sent."
            else:
                content = (
                    f"<error>Unknown tool {name!r}. "
                    "Use execute or send_message.</error>"
                )

            # 3. One tool message per call, linked to the call by its id.
            observations.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
        return observations
