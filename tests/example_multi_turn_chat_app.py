"""
Simple, general-purpose AI chat app using LiteLLM.

This module exposes a small helper class `LiteLLMChatApp` that allows
consumers to generate a reply given a user message and a conversation
history, so downstream users can maintain state across turns.

Environment:
- TEST_APP_MODEL (required): Fully-qualified LiteLLM model identifier
  (e.g., "openai/gpt-4o-mini").
  The system prompt is hardcoded in this file.
"""

import os
import asyncio
from typing import Dict, List

from dotenv import load_dotenv
from litellm import acompletion


# Load environment variables from a .env file if present
load_dotenv()

# Require TEST_APP_MODEL to be set in the environment
TEST_APP_MODEL = os.getenv("TEST_APP_MODEL")
if not TEST_APP_MODEL:
    raise RuntimeError(
        "TEST_APP_MODEL environment variable is required (e.g., 'openai/gpt-4o-mini')."
    )

# Hardcoded system prompt (do not read from env)
SYSTEM_PROMPT = "You are a helpful, concise assistant for a demo retail chatbot."

class LiteLLMChatApp:
    """
    Minimal chat app wrapper around LiteLLM that keeps conversation state
    as a list of {"role": str, "content": str} messages.

    Example usage:
        app = LiteLLMChatApp()
        history: List[Dict[str, str]] = []
        reply, history = await app.respond("Hello!", history)
    """

    def __init__(self, model: str | None = None, system_prompt: str | None = None):
        self.model = model or TEST_APP_MODEL
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def _build_messages(self, history: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def respond(self, user_message: str, history: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
        """
        Generate a reply given a user message and a conversation history.

        Args:
            user_message: The user's message for this turn.
            history: The conversation history to maintain state across turns.

        Returns:
            A tuple of (assistant_text, updated_history).
        """
        messages = self._build_messages(history, user_message)

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=0.6,
            max_tokens=400,
        )

        assistant_text = response.choices[0].message.content

        # Update conversation history for next turn
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_text})

        return assistant_text, history


async def _demo() -> None:
    """Run a small multi-turn demo to illustrate conversation state is preserved."""
    app = LiteLLMChatApp()
    history: List[Dict[str, str]] = []

    user_1 = "Hi, I'm looking for running shoes."
    print(f"[Demo] User: {user_1}")
    assistant_text, history = await app.respond(user_1, history)
    print(f"[Demo] Assistant: {assistant_text}\n")

    user_2 = "I prefer size 10. Do you have Nike options?"
    print(f"[Demo] User: {user_2}")
    assistant_text, history = await app.respond(user_2, history)
    print(f"[Demo] Assistant: {assistant_text}\n")

    user_3 = "Can you summarize my preferences so far?"
    print(f"[Demo] User: {user_3}")
    assistant_text, history = await app.respond(user_3, history)
    print(f"[Demo] Assistant: {assistant_text}")


if __name__ == "__main__":
    asyncio.run(_demo())


