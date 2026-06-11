"""Three eval metrics for the RAG pipeline: Precision@K, Faithfulness, Answer Relevance."""

import json
import re
import time
from typing import Any, Dict, List

from langchain_core.documents import Document


def retrieval_precision_at_k(
    retrieved: List[Document],
    relevant_keywords: List[str],
) -> float:
    """Fraction of retrieved chunks containing at least one relevant keyword."""
    if not retrieved:
        return 0.0
    hits = sum(
        any(kw.lower() in doc.page_content.lower() for kw in relevant_keywords)
        for doc in retrieved
    )
    return round(hits / len(retrieved), 4)


def _parse_judge_response(response: Any) -> Dict[str, Any]:
    """Extract a JSON object from an LLM judge response."""
    text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"score": 0.0, "reason": f"Could not parse: {text[:100]}"}


_FAITHFULNESS_PROMPT = """\
You are evaluating a RAG system. Assess whether the answer is faithful to the retrieved context.

Question: {question}

Retrieved context:
{context}

Answer to evaluate:
{answer}

Faithfulness: does every factual claim in the answer appear in or follow directly from the context?
- 1.0 = answer only uses facts from the context
- 0.5 = answer mostly uses context but adds minor inferences not stated in context
- 0.0 = answer contains claims that contradict or are absent from the context

Respond in JSON only:
{{"score": <number 0.0-1.0>, "reason": "<one sentence>"}}"""


def faithfulness_score(
    question: str,
    context: str,
    answer: str,
    judge_llm: Any,
) -> Dict[str, Any]:
    """LLM-as-judge: is every claim in the answer grounded in the retrieved context?"""
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question, context=context, answer=answer
    )
    response = judge_llm.invoke(prompt)
    result = _parse_judge_response(response)
    result["score"] = float(result.get("score", 0.0))
    return result


_RELEVANCE_PROMPT = """\
You are evaluating a RAG system. Assess whether the answer addresses the question asked.

Question: {question}

Answer to evaluate:
{answer}

Answer relevance: does the answer directly respond to what was asked?
- 1.0 = answer fully addresses the question
- 0.5 = answer is related but incomplete or partially off-topic
- 0.0 = answer does not address the question at all

Respond in JSON only:
{{"score": <number 0.0-1.0>, "reason": "<one sentence>"}}"""


def answer_relevance_score(
    question: str,
    answer: str,
    judge_llm: Any,
) -> Dict[str, Any]:
    """LLM-as-judge: does the answer address what was asked?"""
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    response = judge_llm.invoke(prompt)
    result = _parse_judge_response(response)
    result["score"] = float(result.get("score", 0.0))
    return result


def run_evals(
    pipeline: Any,
    golden_dataset: List[Dict[str, Any]],
    judge_llm: Any,
    judge_delay: float = 0.0,
) -> Dict[str, Any]:
    """Run all three metrics over the golden dataset and print a summary report.

    judge_delay: seconds to sleep between judge LLM calls (use 6.0 for Gemini Flash
                 free tier to avoid 429 RESOURCE_EXHAUSTED errors).
    """
    from rag.generator import format_docs

    results = []
    for case in golden_dataset:
        q = case["question"]
        retrieved = pipeline.retriever.invoke(q)
        context = format_docs(retrieved)
        answer = pipeline.rag_chain.invoke({"question": q, "context": context})

        precision = retrieval_precision_at_k(retrieved, case["relevant_keywords"])
        faith = faithfulness_score(q, context, answer, judge_llm)
        if judge_delay:
            time.sleep(judge_delay)
        relevance = answer_relevance_score(q, answer, judge_llm)
        if judge_delay:
            time.sleep(judge_delay)

        results.append({
            "id": case["id"],
            "question": q,
            "precision_at_k": precision,
            "faithfulness": faith["score"],
            "faithfulness_reason": faith["reason"],
            "answer_relevance": relevance["score"],
            "relevance_reason": relevance["reason"],
        })

    avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)

    print("\n=== RAG Eval Report ===\n")
    print(f"{'ID':<6} {'Precision@K':>12} {'Faithfulness':>13} {'Relevance':>10}")
    print("-" * 45)
    for r in results:
        print(
            f"{r['id']:<6} {r['precision_at_k']:>12.2f}"
            f" {r['faithfulness']:>13.2f}"
            f" {r['answer_relevance']:>10.2f}"
        )
    print("-" * 45)
    print(f"{'AVG':<6} {avg_precision:>12.2f} {avg_faithfulness:>13.2f} {avg_relevance:>10.2f}")
    print()
    print("Targets:  Precision >= 0.75 | Faithfulness >= 0.80 | Relevance >= 0.80")

    return {
        "results": results,
        "avg_precision_at_k": avg_precision,
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_relevance": avg_relevance,
    }
