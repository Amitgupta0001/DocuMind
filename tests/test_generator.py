"""
Tests for LLM prompt builder and generator logic.
"""

from unittest.mock import MagicMock
from openai import RateLimitError
from backend.chatbot.generator import ChatbotGenerator
from backend.chatbot.prompt import build_system_prompt, format_context_excerpts
from backend.config import FALLBACK_ANSWER


def test_format_context_excerpts():
    chunks = [
        {
            "source": "api.pdf",
            "location": "Page 3",
            "page": 3,
            "text": "Authentication requires a Bearer JWT token in headers.",
            "score": 0.88,
        }
    ]
    formatted = format_context_excerpts(chunks)
    assert "=== DOCUMENT EXCERPT [1] ===" in formatted
    assert "api.pdf" in formatted
    assert "Page/Slide 3" in formatted
    assert "Bearer JWT token" in formatted


def test_prompt_contains_fallback_and_injection_rules():
    chunks = [{"source": "test.txt", "text": "Some text"}]
    prompt = build_system_prompt(chunks)
    assert FALLBACK_ANSWER in prompt
    assert "UNTRUSTED USER DATA" in prompt
    assert "STRICT GROUNDING RULES" in prompt


def test_generator_fallback_on_empty_chunks(mock_llm_generator):
    answer = mock_llm_generator.generate_answer("Any question?", [])
    assert answer == FALLBACK_ANSWER


def test_generator_grounded_answer_success(mock_llm_generator):
    chunks = [{"source": "test.txt", "location": "Doc", "text": "Doc content"}]
    answer = mock_llm_generator.generate_answer("How does it work?", chunks)
    assert "verified test answer" in answer


def test_generator_handles_rate_limit():
    dummy_client = MagicMock()
    # Mock RateLimitError
    dummy_client.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit reached",
        response=MagicMock(status_code=429),
        body={"error": "Rate limit"},
    )
    gen = ChatbotGenerator(client=dummy_client)
    chunks = [{"source": "test.txt", "text": "Content"}]
    ans = gen.generate_answer("Question", chunks)
    assert "rate limit" in ans.lower()
