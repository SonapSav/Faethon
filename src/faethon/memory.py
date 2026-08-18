"""Short conversational memory: the last N exchanges, in RAM only.

Deliberately not persisted. Faethon forgets on restart, which is the right
default for an always-on microphone in a home.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class Memory:
    def __init__(self, max_turns: int = 10) -> None:
        # One turn is a user message plus Faethon's reply.
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        if user and assistant:
            self._turns.append((user, assistant))

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def messages(self, system_prompt: str, user: str) -> list[dict[str, Any]]:
        """Build the chat payload: system, history, then the new message."""
        out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for prev_user, prev_assistant in self._turns:
            out.append({"role": "user", "content": prev_user})
            out.append({"role": "assistant", "content": prev_assistant})
        out.append({"role": "user", "content": user})
        return out
