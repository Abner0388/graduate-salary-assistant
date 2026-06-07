"""
Multi-turn conversation memory with token-aware window management.
Uses tiktoken for token counting (cl100k_base as DeepSeek approximation).
"""
import time
from dataclasses import dataclass, field

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TOKENIZER = None


@dataclass
class Message:
    """A single conversation message with cached token count."""
    role: str           # "system" | "user" | "assistant"
    content: str
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)


def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    if _TOKENIZER is None:
        # Fallback: ~1.5 chars per token for Chinese text
        return max(1, len(text.encode("utf-8")) // 2)
    try:
        return len(_TOKENIZER.encode(text))
    except Exception:
        return max(1, len(text.encode("utf-8")) // 2)


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Count total tokens for an array of messages.
    Includes ~4 tokens overhead per message (role + formatting).
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += count_tokens(content) + 4
    return total


class ChatMemory:
    """
    Manages multi-turn conversation history with token tracking.
    Provides windowed context retrieval fitting within a token budget.
    """

    def __init__(self, max_messages: int = 50):
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.summarized_prefix: str = ""  # compressed old messages

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message to the conversation history.
        Auto-trims if exceeding max_messages.
        """
        token_count = count_tokens(content)
        msg = Message(role=role, content=content, token_count=token_count)
        self.messages.append(msg)

        # Trim oldest messages if over limit
        if len(self.messages) > self.max_messages:
            overflow = len(self.messages) - self.max_messages
            self.messages = self.messages[overflow:]

    def size(self) -> int:
        """Return the current number of messages."""
        return len(self.messages)

    def total_tokens(self) -> int:
        """Return total token count across all messages."""
        return sum(m.token_count for m in self.messages) + len(self.messages) * 4

    def get_context_messages(self, max_tokens: int) -> list[dict]:
        """
        Get conversation messages that fit within max_tokens.
        Prioritizes recent messages. Prepends summarized_prefix if available.

        Returns list of {"role": role, "content": content} dicts ready for API.
        """
        if not self.messages:
            return []

        result = []
        budget = max_tokens

        # If we have a summarized prefix, add it first
        if self.summarized_prefix:
            prefix_tokens = count_tokens(self.summarized_prefix)
            if prefix_tokens <= budget - 50:
                result.append({
                    "role": "system",
                    "content": f"[对话摘要] {self.summarized_prefix}"
                })
                budget -= prefix_tokens

        # Add messages from newest to oldest until budget exhausted
        selected = []
        remaining = budget
        for msg in reversed(self.messages):
            needed = msg.token_count + 4
            if needed <= remaining:
                selected.append(msg)
                remaining -= needed
            else:
                break

        # Reverse back to chronological order
        for msg in reversed(selected):
            result.append({"role": msg.role, "content": msg.content})

        return result

    def summarize(self, summarizer_fn) -> None:
        """
        Compress oldest 50% of messages into a summary prefix.
        Calls summarizer_fn(messages_text) -> summary_string.
        Keeps the 5 most recent messages intact.
        """
        if len(self.messages) <= 10:
            return

        # Split: oldest half gets compressed, newest 5 stay
        split = max(5, len(self.messages) // 2)
        old_msgs = self.messages[:split]
        recent = self.messages[-5:]

        # Build text for summarization
        text = "\n".join(
            f"[{m.role}]: {m.content[:200]}"
            for m in old_msgs
        )

        try:
            summary = summarizer_fn(text)
            if summary and len(summary) > 10:
                self.summarized_prefix = summary[:500]
        except Exception:
            # Summarization failed — just drop oldest messages
            pass

        # Keep only recent messages
        self.messages = recent

    def clear(self) -> None:
        """Reset all memory."""
        self.messages.clear()
        self.summarized_prefix = ""
