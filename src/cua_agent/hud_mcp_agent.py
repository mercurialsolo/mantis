"""Gemma4 MCPAgent — wraps our LlamaCppBrain for HUD OSWorld evaluation.

Subclasses hud.agents.MCPAgent so that `run_dataset` runs *our* agent logic
(system prompt, multi-frame reasoning, action history, loop detection) instead
of HUD's generic OpenAI-compatible wrapper.

Usage:
    from hud.datasets import run_dataset
    from cua_agent.hud_mcp_agent import Gemma4MCPAgent

    results = await run_dataset(
        name="gemma4-osworld",
        dataset="hud-evals/OSWorld-Verified",
        agent_class=Gemma4MCPAgent,
        agent_config={
            "base_url": "http://localhost:8080/v1",
            "model": "gemma-4",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        max_steps=15,
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import mcp.types as types
import requests

from hud.agents.base import MCPAgent
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .brain_llamacpp import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# How many recent screenshots to keep for temporal context.
DEFAULT_SCREENSHOT_WINDOW = 3


class Gemma4MCPAgent(MCPAgent):
    """MCP agent powered by Gemma4 via llama.cpp (OpenAI-compatible API).

    Adds our custom capabilities on top of HUD's agent loop:
    - Our tuned system prompt with temporal-context instructions
    - Sliding window of recent screenshots for multi-frame reasoning
    - Action history tracking for loop awareness
    - OpenAI function-calling via llama-server
    """

    metadata: ClassVar[dict[str, Any]] = {
        "agent": "gemma4-cua",
        "framework": "cua-agent",
    }

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080/v1",
        model: str = "gemma-4",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        screenshot_window: int = DEFAULT_SCREENSHOT_WINDOW,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            model_name=model,
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._screenshot_window = screenshot_window

        # Sliding window of recent screenshots (base64 PNGs)
        self._recent_screenshots: list[str] = []
        # Action history for loop-awareness context
        self._action_history: list[str] = []

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def get_system_messages(self) -> list[dict[str, Any]]:
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}]
        return []

    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Send messages to llama-server, parse the response."""
        tools = self._build_openai_tools()

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("llama-server request failed: %s", e)
            return AgentResponse(
                content=f"Error calling model: {e}",
                tool_calls=[],
                done=True,
                isError=True,
            )

        return self._parse_chat_response(data)

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[dict[str, Any]]:
        """Convert MCP ContentBlocks → OpenAI user message.

        Images are stored in a sliding window so the model gets temporal
        context (previous screenshots alongside the current one).
        """
        content_parts: list[dict[str, Any]] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                content_parts.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                self._push_screenshot(block.data)

                # Rebuild full screenshot window with temporal labels
                content_parts = [
                    p for p in content_parts
                    if p.get("type") != "image_url"
                    and not (p.get("type") == "text" and p.get("text", "").startswith("[Frame "))
                ]
                n = len(self._recent_screenshots)
                for i, shot in enumerate(self._recent_screenshots):
                    label = "CURRENT" if i == n - 1 else f"t-{n - 1 - i}"
                    content_parts.append({"type": "text", "text": f"[Frame {label}]"})
                    mime = block.mimeType or "image/png"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{shot}"},
                    })
            elif isinstance(block, types.EmbeddedResource):
                if hasattr(block.resource, "text"):
                    content_parts.append({"type": "text", "text": block.resource.text})

        # Inject action history for loop awareness
        if self._action_history:
            recent = self._action_history[-10:]
            history_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(recent))
            content_parts.append({"type": "text", "text": f"Recent actions:\n{history_str}"})

        if not content_parts:
            return []
        return [{"role": "user", "content": content_parts}]

    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[dict[str, Any]]:
        """Convert MCP tool results → OpenAI messages.

        Emits the assistant tool_calls message + tool result messages.
        Also captures screenshots from tool results and tracks action history.
        """
        # Emit assistant message with tool_calls for well-formed conversation
        assistant_tool_calls = []
        for tc in tool_calls:
            assistant_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments or {}),
                },
            })
            self._action_history.append(f"{tc.name}({json.dumps(tc.arguments or {})})")

        rendered: list[dict[str, Any]] = [
            {"role": "assistant", "tool_calls": assistant_tool_calls}
        ]

        image_parts: list[dict[str, Any]] = []

        for tc, result in zip(tool_calls, tool_results, strict=False):
            text_parts: list[str] = []
            if result.structuredContent:
                text_parts.append(json.dumps(result.structuredContent))
            elif result.content:
                for item in result.content:
                    if isinstance(item, types.TextContent):
                        text_parts.append(item.text)
                    elif isinstance(item, types.ImageContent):
                        self._push_screenshot(item.data)
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                        })

            rendered.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "".join(text_parts) if text_parts else "Tool executed successfully",
            })

        # Send screenshots as user message with temporal labels
        if image_parts:
            parts: list[dict[str, Any]] = [
                {"type": "text", "text": "Tool returned the following:"},
            ]
            n = len(self._recent_screenshots)
            for i, shot in enumerate(self._recent_screenshots):
                label = "CURRENT" if i == n - 1 else f"t-{n - 1 - i}"
                parts.append({"type": "text", "text": f"[Frame {label}]"})
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{shot}"},
                })
            rendered.append({"role": "user", "content": parts})

        return rendered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        """Convert HUD's MCP tool schemas to OpenAI function-calling format."""
        tools = []
        for tool in self.get_available_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            })
        return tools

    def _push_screenshot(self, b64_data: str) -> None:
        """Add a screenshot to the sliding window."""
        self._recent_screenshots.append(b64_data)
        if len(self._recent_screenshots) > self._screenshot_window:
            self._recent_screenshots = self._recent_screenshots[-self._screenshot_window:]

    def _parse_chat_response(self, data: dict) -> AgentResponse:
        """Parse OpenAI chat completion → AgentResponse."""
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        reasoning = message.get("reasoning_content") or message.get("thinking") or ""
        content = message.get("content") or ""

        # Parse tool calls
        tool_calls: list[MCPToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(MCPToolCall(
                id=tc.get("id", ""),
                name=name,
                arguments=arguments,
            ))

        is_done = (
            finish_reason == "stop"
            or (not tool_calls and finish_reason != "tool_calls")
        )

        return AgentResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            done=is_done,
            raw=data,
        )
