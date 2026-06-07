"""
Context Manager — orchestrates token budgeting, message assembly,
and auto-summarization for LLM API calls.
"""
from .config import MAX_CONTEXT_TOKENS
from .memory.chat_memory import ChatMemory, count_tokens, count_messages_tokens
from .memory.session_memory import SessionMemory


# Token budget allocation (percentages of MAX_CONTEXT_TOKENS)
BUDGET_SYSTEM = 0.12      # system prompt
BUDGET_RAG = 0.15          # RAG retrieved context
BUDGET_PROFILE = 0.05      # student profile context
BUDGET_HISTORY = 0.45      # conversation history
BUDGET_USER = 0.18          # current user message
BUDGET_OVERHEAD = 0.05      # API overhead


class ContextManager:
    """
    Assembles the full messages array for LLM API calls,
    respecting the configured token budget.
    """

    def __init__(
        self,
        chat_memory: ChatMemory,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ):
        self.chat_memory = chat_memory
        self.max_tokens = max_context_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return count_tokens(text)

    def _allocate(self) -> dict:
        """Return token budget allocation per section."""
        t = self.max_tokens
        return {
            "system": int(t * BUDGET_SYSTEM),
            "rag": int(t * BUDGET_RAG),
            "profile": int(t * BUDGET_PROFILE),
            "history": int(t * BUDGET_HISTORY),
            "user": int(t * BUDGET_USER),
        }

    def build_messages(
        self,
        system_prompt: str,
        user_message: str,
        rag_context: str = "",
        profile_context: str = "",
    ) -> list[dict]:
        """
        Assemble the complete messages payload for the LLM API.

        Structure:
        1. System message (system_prompt + rag_context + profile_context)
        2. Summarized history prefix (if any)
        3. Recent conversation messages (within remaining budget)
        4. Current user message

        Returns list of {"role": role, "content": content}.
        """
        budget = self._allocate()
        messages = []

        # 1. System message with injected context
        system_content = system_prompt
        if rag_context:
            rag_part = rag_context[:budget["rag"] * 2]  # allow flexible sizing
            system_content = f"{system_content}\n\n{rag_part}"
        if profile_context:
            profile_part = profile_context[:budget["profile"] * 2]
            system_content = f"{system_content}\n\n{profile_part}"

        system_tokens = count_tokens(system_content)
        if system_tokens > budget["system"] * 2:
            # Truncate rag/profile context to fit
            base_tokens = count_tokens(system_prompt)
            remain = budget["system"] * 2 - base_tokens
            if rag_context and profile_context:
                system_content = system_prompt + "\n\n" + rag_context[:remain // 2] + "\n" + profile_context[:remain // 2]
            elif rag_context:
                system_content = system_prompt + "\n\n" + rag_context[:remain]
            else:
                system_content = system_prompt

        messages.append({"role": "system", "content": system_content})

        # 2. Conversation history (within budget)
        history_tokens = budget["history"]
        history_msgs = self.chat_memory.get_context_messages(history_tokens)
        for hm in history_msgs:
            messages.append(hm)

        # 3. Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def maybe_summarize(self, summarizer_fn) -> bool:
        """
        Check if chat history is consuming too much of the token budget.
        If so, trigger summarization. Returns True if summarization happened.
        """
        chat_tokens = self.chat_memory.total_tokens()
        history_budget = self._allocate()["history"]

        if chat_tokens > history_budget:
            self.chat_memory.summarize(summarizer_fn)
            return True
        return False

    def inject_profile(self, student: dict | None) -> str:
        """Format a student profile as compact context for the system prompt."""
        return SessionMemory.profile_to_context(student)

    def inject_rag(
        self,
        similar_profiles: list[dict],
        corpus: dict,
    ) -> str:
        """Format RAG results as compact context."""
        from .rag.retriever import format_retrieved_context
        return format_retrieved_context(similar_profiles, corpus)

    def get_budget_usage(self, messages: list[dict]) -> int:
        """Return total token usage for the given messages."""
        return count_messages_tokens(messages)
