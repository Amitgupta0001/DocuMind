"""
DocuMind Chatbot Generator.
Orchestrates grounded answer generation via OpenRouter or OpenAI with
multi-provider fallback, timeout resilience, and comprehensive error handling.
"""

import logging
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from backend.chatbot.prompt import build_system_prompt
from backend.config import (
    FALLBACK_ANSWER,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger("documind.generator")


class ChatbotGenerator:
    """
    LLM Answer Generator with dual-provider support (OpenRouter / OpenAI).
    Configurable, timeout-guarded, and test-mockable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        self._custom_client = client
        self.provider = LLM_PROVIDER

        if client is not None:
            self.client = client
            self.model = model or "mock-model"
            logger.info("ChatbotGenerator initialized with injected custom client.")
            return

        # Determine credentials
        if self.provider == "openai" or (not OPENROUTER_API_KEY and OPENAI_API_KEY):
            self.provider = "openai"
            self.api_key = api_key or OPENAI_API_KEY
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or OPENAI_MODEL
        else:
            self.provider = "openrouter"
            self.api_key = api_key or OPENROUTER_API_KEY
            self.base_url = base_url or OPENROUTER_BASE_URL
            self.model = model or OPENROUTER_MODEL

        if not self.api_key:
            logger.warning(
                "No LLM API key detected. Answers will return fallback response until configured."
            )
            self.client = None
        else:
            headers = {}
            if self.provider == "openrouter":
                headers = {
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "DocuMind Documentation Chatbot",
                }

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=float(REQUEST_TIMEOUT_SECONDS),
                default_headers=headers,
            )
            logger.info(
                "LLM generator initialized: provider=%s, model=%s, base_url=%s",
                self.provider,
                self.model,
                self.base_url,
            )

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate grounded response from retrieved documentation context.
        """
        if not chunks:
            logger.info("Empty context provided. Returning fallback answer.")
            return FALLBACK_ANSWER

        if self.client is None:
            logger.warning("LLM client not configured (missing API key). Returning fallback.")
            return FALLBACK_ANSWER

        system_prompt = build_system_prompt(chunks=chunks)
        logger.info(
            "Requesting completion from %s (%s) for query: '%s'",
            self.provider,
            self.model,
            question[:50],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question.strip()},
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                logger.warning("LLM returned an empty response. Returning fallback.")
                return FALLBACK_ANSWER

            cleaned_answer = answer.strip()
            logger.info("Answer generated successfully (%d characters).", len(cleaned_answer))
            return cleaned_answer

        except AuthenticationError as exc:
            logger.error("LLM authentication failure: %s", exc)
            return (
                "AI service authentication failed. Please verify your API key in the environment."
            )

        except RateLimitError as exc:
            logger.error("LLM rate limit / quota exceeded: %s", exc)
            return (
                "AI service rate limit or quota exceeded. Please check credits or try again shortly."
            )

        except APIConnectionError as exc:
            logger.error("Network connection error to LLM provider: %s", exc)
            return (
                "Could not connect to the AI service. Please check network connection."
            )

        except APIError as exc:
            logger.error("LLM provider API error: %s", exc)
            return f"The AI service returned an error ({exc.code or 'unknown'}). Please retry."

        except Exception as exc:
            logger.exception("Unexpected error in generator: %s", exc)
            return "An unexpected error occurred while generating the answer."


_generator: Optional[ChatbotGenerator] = None


def get_generator() -> ChatbotGenerator:
    """Lazy singleton for generator."""
    global _generator
    if _generator is None:
        _generator = ChatbotGenerator()
    return _generator


def set_generator(generator: Optional[ChatbotGenerator]) -> None:
    """Inject custom or mock generator for unit tests."""
    global _generator
    _generator = generator


def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """Module-level entry point."""
    gen = get_generator()
    return gen.generate_answer(question=question, chunks=chunks)