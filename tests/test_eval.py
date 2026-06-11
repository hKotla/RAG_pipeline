"""Unit tests for rag/eval.py — no API calls, LLM is mocked."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.eval import (
    _parse_judge_response,
    answer_relevance_score,
    faithfulness_score,
    retrieval_precision_at_k,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _doc(text: str) -> Document:
    return Document(page_content=text)


def _mock_llm(json_str: str):
    """Return a mock LLM whose .invoke() returns a response with .content = json_str."""
    llm = MagicMock()
    response = MagicMock()
    response.content = json_str
    llm.invoke.return_value = response
    return llm


# ── retrieval_precision_at_k ─────────────────────────────────────────────────

def test_precision_all_relevant():
    docs = [_doc("crop yield drops due to drought"), _doc("farmers face monsoon failure")]
    assert retrieval_precision_at_k(docs, ["crop yield", "monsoon"]) == 1.0


def test_precision_half_relevant():
    docs = [_doc("solar energy in Rajasthan"), _doc("monsoon rainfall patterns")]
    assert retrieval_precision_at_k(docs, ["monsoon", "rainfall"]) == 0.5


def test_precision_none_relevant():
    docs = [_doc("completely unrelated text"), _doc("nothing about the topic")]
    assert retrieval_precision_at_k(docs, ["glacier", "Himalayan"]) == 0.0


def test_precision_empty_retrieved():
    assert retrieval_precision_at_k([], ["anything"]) == 0.0


def test_precision_case_insensitive():
    docs = [_doc("MONSOON season brings heavy RAINFALL")]
    assert retrieval_precision_at_k(docs, ["monsoon"]) == 1.0


def test_precision_partial_keyword_match():
    # "crop" should not match "crop yield" keyword because we match keyword IN text,
    # not the other way — "crop yield" IS in "reduced crop yield figures"
    docs = [_doc("reduced crop yield figures for 2023")]
    assert retrieval_precision_at_k(docs, ["crop yield"]) == 1.0


# ── _parse_judge_response ────────────────────────────────────────────────────

def test_parse_with_content_attr():
    response = MagicMock()
    response.content = '{"score": 0.8, "reason": "mostly faithful"}'
    result = _parse_judge_response(response)
    assert result["score"] == 0.8
    assert "faithful" in result["reason"]


def test_parse_plain_string():
    result = _parse_judge_response('{"score": 0.5, "reason": "partial"}')
    assert result["score"] == 0.5
    assert result["reason"] == "partial"


def test_parse_unparseable_returns_zero():
    response = MagicMock()
    response.content = "sorry I cannot score this"
    result = _parse_judge_response(response)
    assert result["score"] == 0.0
    assert "Could not parse" in result["reason"]


def test_parse_json_embedded_in_prose():
    response = MagicMock()
    response.content = 'Sure, here is my rating: {"score": 0.9, "reason": "all claims supported"} Done.'
    result = _parse_judge_response(response)
    assert result["score"] == 0.9


# ── faithfulness_score ───────────────────────────────────────────────────────

def test_faithfulness_high_score():
    llm = _mock_llm('{"score": 0.9, "reason": "all claims supported by context"}')
    result = faithfulness_score("q?", "context text", "answer text", llm)
    assert result["score"] == 0.9
    assert isinstance(result["reason"], str)


def test_faithfulness_low_score():
    llm = _mock_llm('{"score": 0.1, "reason": "answer hallucinated facts"}')
    result = faithfulness_score("q?", "context", "made up answer", llm)
    assert result["score"] == 0.1


def test_faithfulness_score_coerced_to_float():
    llm = _mock_llm('{"score": 1, "reason": "perfect"}')
    result = faithfulness_score("q?", "ctx", "ans", llm)
    assert isinstance(result["score"], float)
    assert result["score"] == 1.0


def test_faithfulness_calls_llm_once():
    llm = _mock_llm('{"score": 0.7, "reason": "ok"}')
    faithfulness_score("q?", "ctx", "ans", llm)
    llm.invoke.assert_called_once()


# ── answer_relevance_score ───────────────────────────────────────────────────

def test_relevance_high_score():
    llm = _mock_llm('{"score": 1.0, "reason": "directly answers the question"}')
    result = answer_relevance_score("What is X?", "X is Y.", llm)
    assert result["score"] == 1.0


def test_relevance_off_topic():
    llm = _mock_llm('{"score": 0.0, "reason": "does not address the question"}')
    result = answer_relevance_score("What is X?", "The weather is nice.", llm)
    assert result["score"] == 0.0


def test_relevance_score_coerced_to_float():
    llm = _mock_llm('{"score": 0, "reason": "no relevance"}')
    result = answer_relevance_score("q?", "ans", llm)
    assert isinstance(result["score"], float)


def test_relevance_calls_llm_once():
    llm = _mock_llm('{"score": 0.5, "reason": "partial"}')
    answer_relevance_score("q?", "ans", llm)
    llm.invoke.assert_called_once()
