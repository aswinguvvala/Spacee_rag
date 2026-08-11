"""LLM-backed answer generation with mandatory per-sentence chunk citations.

Provider abstraction
---------------------
``LLMClient`` is the one interface every provider must implement
(``generate_structured``). Nothing else in this module -- or in
``app.py`` -- talks to a specific vendor SDK. Swapping providers means
writing one new ``LLMClient`` subclass and pointing ``LLM_PROVIDER`` at it;
no other file changes. The default implementation, ``AnthropicClient``, uses
Claude's tool-use feature (a forced tool call) to get back parsed, validated
JSON instead of free text that would otherwise need to be regexed apart.

Abstention is a prompt-design problem, not a post-hoc filter
--------------------------------------------------------------
The system prompt requires the model to check, before writing anything,
whether the provided source chunks actually support an answer. If they
don't, it must set ``abstained=true`` and explain why instead of answering
from outside knowledge. This is enforced by the tool schema itself (an
``abstained`` boolean field the model must fill in) rather than by scanning
the generated text afterwards for hedging language.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.retrieval import RetrievalResult
from src.utils import get_logger

load_dotenv()

logger = get_logger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 4096
REQUEST_TIMEOUT_SECONDS = 60

TOOL_NAME = "provide_grounded_answer"

TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "abstained": {
            "type": "boolean",
            "description": (
                "true if the provided source chunks do not contain enough "
                "information to answer the question -- even partially. "
                "false only if every sentence below can be backed by a cited chunk."
            ),
        },
        "abstain_reason": {
            "type": "string",
            "description": (
                "If abstained is true, a one-sentence explanation of what "
                "information is missing from the sources. Empty string otherwise."
            ),
        },
        "sentences": {
            "type": "array",
            "description": "The answer, split into sentences. Empty if abstained is true.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "One sentence of the answer."},
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "chunk_id(s) this sentence is directly drawn from. Must be ids from the provided sources -- never invent one.",
                    },
                },
                "required": ["text", "chunk_ids"],
            },
        },
    },
    "required": ["abstained", "abstain_reason", "sentences"],
}

SYSTEM_PROMPT = """You are a careful research assistant. You answer questions using ONLY the \
numbered source chunks provided in the user message, never your own outside knowledge, even if \
you happen to know the answer.

Rules:
1. Every sentence in your answer must be tagged with the chunk_id(s) of the source chunk(s) it \
is directly drawn from. Only cite chunk_ids that literally appear in the provided sources -- \
never invent one.
2. Do not merge facts from two different chunks into one sentence unless you cite both chunks \
for that sentence.
3. Before writing anything, check whether the provided sources actually contain the information \
needed to answer the question -- even partially, even if the topic sounds related. If they do \
not, do not guess, extrapolate, or fall back on general knowledge. Instead set abstained=true, \
leave sentences empty, and give a one-sentence abstain_reason naming what's missing.
4. Only set abstained=false when every sentence you write can be backed by a specific cited chunk.

You must always respond by calling the provide_grounded_answer tool -- never respond with plain text."""


class GenerationError(Exception):
    """Raised when the LLM call fails or returns a response that violates the contract."""


class SentenceCitation(BaseModel):
    """One sentence of a generated answer and the chunk id(s) it draws from."""

    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """A full generated answer: either grounded sentences, or an abstention.

    Attributes:
        abstained: True if the model declined to answer because the
            retrieved sources didn't support it.
        sentences: The answer's sentences with citations. Empty when
            ``abstained`` is True.
        abstain_reason: Explanation for the abstention. ``None`` when
            ``abstained`` is False.
    """

    abstained: bool
    sentences: list[SentenceCitation] = Field(default_factory=list)
    abstain_reason: str | None = None


class LLMClient(ABC):
    """Provider-agnostic interface for structured, tool-forced generation."""

    @abstractmethod
    def generate_structured(
        self, system: str, user: str, tool_schema: dict, tool_name: str
    ) -> dict:
        """Call the LLM and return the parsed input of a forced tool call.

        Args:
            system: System prompt.
            user: User-turn content.
            tool_schema: JSON schema for the forced tool's input.
            tool_name: Name of the tool the model must call.

        Returns:
            The tool call's parsed ``input`` as a plain dict.

        Raises:
            GenerationError: If the call fails or the model doesn't return
                the expected tool call.
        """
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """LLMClient implementation backed by the Anthropic Messages API."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the client from environment variables.

        Args:
            model: Anthropic model id. Defaults to the ``ANTHROPIC_MODEL``
                env var, then :data:`DEFAULT_ANTHROPIC_MODEL`.

        Raises:
            GenerationError: If ``ANTHROPIC_API_KEY`` is not set or the SDK
                fails to initialize.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise GenerationError(f"anthropic package not installed: {exc}") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise GenerationError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    def generate_structured(
        self, system: str, user: str, tool_schema: dict, tool_name: str
    ) -> dict:
        tools = [
            {
                "name": tool_name,
                "description": "Provide a grounded answer with per-sentence citations, or abstain.",
                "input_schema": tool_schema,
            }
        ]
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=tools,
                tool_choice={"type": "tool", "name": tool_name},
            )
        except self._anthropic.AuthenticationError as exc:
            raise GenerationError(f"Anthropic authentication failed: {exc}") from exc
        except self._anthropic.RateLimitError as exc:
            raise GenerationError(f"Anthropic rate limit exceeded: {exc}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise GenerationError(f"Could not connect to Anthropic API: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise GenerationError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input

        raise GenerationError(
            f"Model response did not include a call to {tool_name!r} (stop_reason={response.stop_reason!r})"
        )


class OpenRouterClient(LLMClient):
    """LLMClient implementation backed by OpenRouter's OpenAI-compatible API.

    Included alongside :class:`AnthropicClient` so the demo can run entirely
    on OpenRouter's free-tier models (no payment method required) --
    swapping providers is a one-line env var change (``LLM_PROVIDER``), not
    a code change, since both implement the same :class:`LLMClient` interface.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize the client from environment variables.

        Args:
            model: OpenRouter model id (e.g. ``"openai/gpt-oss-20b:free"``).
                Defaults to the ``OPENROUTER_MODEL`` env var, then
                :data:`DEFAULT_OPENROUTER_MODEL`.

        Raises:
            GenerationError: If ``OPENROUTER_API_KEY`` is not set.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise GenerationError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free, no payment method needed, from https://openrouter.ai/keys)."
            )
        self.api_key = api_key
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

    def generate_structured(
        self, system: str, user: str, tool_schema: dict, tool_name: str
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Provide a grounded answer with per-sentence citations, or abstain.",
                        "parameters": tool_schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        }
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise GenerationError(f"OpenRouter request timed out: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            raise GenerationError(f"OpenRouter API error ({response.status_code}): {response.text}") from exc
        except requests.exceptions.RequestException as exc:
            raise GenerationError(f"Could not connect to OpenRouter API: {exc}") from exc

        data = response.json()
        try:
            tool_calls = data["choices"][0]["message"].get("tool_calls") or []
        except (KeyError, IndexError) as exc:
            raise GenerationError(f"Unexpected OpenRouter response shape: {exc}; raw={data}") from exc

        for call in tool_calls:
            if call.get("function", {}).get("name") == tool_name:
                try:
                    return json.loads(call["function"]["arguments"])
                except json.JSONDecodeError as exc:
                    raise GenerationError(f"OpenRouter tool call arguments were not valid JSON: {exc}") from exc

        raise GenerationError(f"Model response did not include a call to {tool_name!r}; raw={data}")


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory returning the configured :class:`LLMClient`.

    Args:
        provider: Provider name. Defaults to the ``LLM_PROVIDER`` env var,
            then ``"anthropic"``.

    Returns:
        An initialized :class:`LLMClient`.

    Raises:
        GenerationError: If ``provider`` is not a supported value.
    """
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openrouter":
        return OpenRouterClient()
    raise GenerationError(
        f"Unsupported LLM_PROVIDER {provider!r}. Implement an LLMClient subclass to add one."
    )


def _format_sources(retrieved: list[RetrievalResult]) -> str:
    blocks = []
    for result in retrieved:
        blocks.append(f"[chunk_id: {result.chunk.chunk_id}]\n{result.chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    retrieved: list[RetrievalResult],
    client: LLMClient | None = None,
) -> GeneratedAnswer:
    """Generate a grounded, cited answer -- or an abstention -- for ``question``.

    Args:
        question: The user's natural-language question.
        retrieved: Chunks returned by :class:`~src.retrieval.Retriever` for
            this question. If empty, the LLM is not called at all: there is
            nothing to cite, so the function abstains directly.
        client: An :class:`LLMClient` to use. Defaults to
            :func:`get_llm_client`'s result.

    Returns:
        A validated :class:`GeneratedAnswer`.

    Raises:
        GenerationError: If the LLM call fails or its response can't be
            validated against the expected schema.
    """
    if not retrieved:
        logger.info("No chunks retrieved for %r; abstaining without an LLM call", question)
        return GeneratedAnswer(
            abstained=True,
            sentences=[],
            abstain_reason="No source chunks were retrieved for this question.",
        )

    client = client or get_llm_client()
    user_prompt = f"Question: {question}\n\nSource chunks:\n{_format_sources(retrieved)}"

    raw = client.generate_structured(SYSTEM_PROMPT, user_prompt, TOOL_SCHEMA, TOOL_NAME)

    try:
        answer = GeneratedAnswer.model_validate(raw)
    except ValidationError as exc:
        raise GenerationError(f"Model response failed schema validation: {exc}") from exc

    if not answer.abstained and not answer.sentences:
        raise GenerationError("Model claimed abstained=false but returned zero sentences")

    logger.info(
        "Generated answer for %r: abstained=%s, %d sentence(s)",
        question,
        answer.abstained,
        len(answer.sentences),
    )
    return answer


if __name__ == "__main__":
    import sys

    from src.retrieval import Retriever

    question = " ".join(sys.argv[1:]) or "Who commanded Apollo 13?"
    retriever = Retriever()
    results = retriever.retrieve(question, k=5)
    answer = generate_answer(question, results)

    print(f"Question: {question}")
    if answer.abstained:
        print(f"ABSTAINED: {answer.abstain_reason}")
    else:
        for s in answer.sentences:
            print(f"- {s.text}  {s.chunk_ids}")
