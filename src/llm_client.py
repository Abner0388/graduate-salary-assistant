"""
DeepSeek LLM client wrapper (OpenAI-compatible API).
"""
import time
import openai

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
)


def _get_client() -> openai.OpenAI:
    """Create a configured OpenAI client pointing to DeepSeek API."""
    return openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_TIMEOUT,
    )


def ask_deepseek(
    system_prompt: str,
    user_message: str,
    temperature: float = LLM_TEMPERATURE,
    model: str | None = None,
) -> str:
    """
    Send a prompt to DeepSeek and return the response text.

    Args:
        system_prompt: System-level instruction.
        user_message: User query / context.
        temperature: Sampling temperature (0-1).
        model: Override the default model name.

    Returns:
        The LLM response text, or an error message if the API call fails.
    """
    client = _get_client()
    model_name = model or DEEPSEEK_MODEL

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()

        except openai.APITimeoutError:
            if attempt < LLM_MAX_RETRIES - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return "[AI 回复暂时不可用：API 请求超时]"

        except openai.RateLimitError:
            if attempt < LLM_MAX_RETRIES - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return "[AI 回复暂时不可用：API 请求频率过高，请稍后再试]"

        except openai.AuthenticationError:
            return "[API 密钥无效，请检查 DeepSeek API Key 配置]"

        except Exception as e:
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return f"[AI 回复暂时不可用：{str(e)[:100]}]"

    return "[AI 回复暂时不可用]"


def ask_deepseek_stream(
    system_prompt: str,
    user_message: str,
    temperature: float = LLM_TEMPERATURE,
    model: str | None = None,
):
    """
    Stream a response from DeepSeek. Yields text chunks.
    """
    client = _get_client()
    model_name = model or DEEPSEEK_MODEL

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=800,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        yield "[AI 回复暂时不可用]"
