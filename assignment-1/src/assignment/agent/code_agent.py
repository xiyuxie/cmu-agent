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
        # When the agent is given skills (skills_path), Agent.__init__ has
        # already appended INVOKE_SKILL_TOOL to self.tools before this line
        # runs. `*self.tools` keeps that tool in the list instead of
        # overwriting it.
        self.tools = [EXECUTE_TOOL, SEND_MESSAGE_TOOL, *self.tools]

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        # The four values describe the sandbox machine, not the machine this
        # code runs on. The environment collected them when the sandbox started.
        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        # Standing instructions: sent as the `system` message on every request.
        self.system_prompt = (
            "You are a software engineer resolving an issue in a code repository. "
            f"The repository is checked out at {self.env.cwd} on the machine "
            "described below.\n"
            "\n"
            "You can only act through your tools. Use `execute` to run shell "
            "commands: read the code, run it, edit files, and run tests. Make a "
            "tool call in every response; a response without one does nothing.\n"
            "\n"
            "Work in this order:\n"
            "1. Read the issue and explore the relevant code.\n"
            "2. Reproduce the problem before changing anything, so you know "
            "what failing looks like.\n"
            "3. Make the smallest change to the source code that fixes the root "
            "cause. Do not edit or delete existing tests to make them pass.\n"
            "4. Verify the fix: run your reproduction again, then the existing "
            "tests, and confirm nothing else broke.\n"
            "\n"
            "If a command fails or a tool returns an error, read the message and "
            "try a different approach. When the issue is resolved and verified, "
            "call `send_message` with a short summary of what you changed and "
            "how you checked it. That ends your work, so call it only once you "
            "are done.\n"
            "\n"
            f"<system_information>\n{system_information}\n</system_information>"
        )
        # The job itself: sent as the first `user` message.
        self.task_prompt = (
            f"Resolve the following issue in the repository at {self.env.cwd}.\n"
            "\n"
            f"<issue>\n{self.task}\n</issue>"
        )
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
