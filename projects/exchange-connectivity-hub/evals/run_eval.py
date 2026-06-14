"""RAGAS evaluation harness for the RAG system.

This script loads the golden dataset, runs the RAG chain over all questions,
and evaluates the results using RAGAS metrics.
"""

# Compatibility shim for ragas + langchain-community 0.4.x
# ragas imports from old langchain_community.chat_models.vertexai path
import sys
from types import ModuleType

# Create a shim module that proxies the actual ChatVertexAI import
_shim_module = ModuleType("langchain_community.chat_models.vertexai")


# Lazy import function
def _get_chat_vertex_ai():
    try:
        from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI

        return _ChatVertexAI
    except ModuleNotFoundError:
        # langchain-google-vertexai is an optional, heavyweight dependency we do
        # not install: this project evaluates with the Anthropic provider, so
        # ragas never instantiates ChatVertexAI. It only needs the symbol to
        # resolve at import time, so a placeholder class is sufficient.
        class _ChatVertexAIUnavailable:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError(
                    "ChatVertexAI is unavailable: install langchain-google-vertexai "
                    "to use the Vertex AI provider."
                )

        return _ChatVertexAIUnavailable


# Make the module act like a proxy for ChatVertexAI
class _VertexAIProxy:
    """Proxy that lazily imports ChatVertexAI."""

    def __getattr__(self, name: str):
        if name == "ChatVertexAI":
            return _get_chat_vertex_ai()
        raise AttributeError(f"'module' has no attribute '{name}'")


# Set up the shim
sys.modules["langchain_community.chat_models.vertexai"] = _VertexAIProxy()
sys.modules["langchain_community.chat_models"] = _VertexAIProxy()

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from datasets import Dataset  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_community.embeddings import VoyageEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from exchange_connectivity_hub.config import get_anthropic_api_key, get_voyage_api_key  # noqa: E402
from exchange_connectivity_hub.retrieval.rag_chain import (  # noqa: E402
    create_rag_chain,
)


# Refusal phrases that indicate the model correctly refused to answer
REFUSAL_PHRASES = [
    "I don't have enough information",
    "I don't have enough information to answer",
    "not enough information",
    "cannot be determined from the context",
    "the context does not contain",
    "not mentioned in the context",
    "cannot answer from the given context",
]


def is_refusal(answer: str) -> bool:
    """Check if an answer indicates a refusal due to lack of information."""
    answer_lower = answer.lower()
    return any(phrase.lower() in answer_lower for phrase in REFUSAL_PHRASES)


def load_golden_dataset(path: Path) -> tuple[list[dict], list[dict]]:
    """Load the golden dataset and split into answerable and unanswerable.

    Args:
        path: Path to golden_dataset.json

    Returns:
        Tuple of (answerable_questions, unanswerable_questions)
    """
    with open(path) as f:
        data = json.load(f)

    return data["answerable"], data["unanswerable"]


def extract_contexts(sources: list[dict]) -> list[str]:
    """Extract context strings from source metadata.

    Since we only have source metadata (not full content), we construct
    minimal context strings that reference the source.
    """
    contexts = []
    for source in sources:
        filename = source.get("filename", "Unknown")
        page = source.get("page_number", "?")
        exchange = source.get("exchange", "Unknown")
        contexts.append(f"[Source: {filename}, page {page}, exchange: {exchange}]")
    return contexts


def run_evaluation(
    evals_dir: Path,
    exchanges: list[str] | None = None,
) -> dict:
    """Run the full evaluation pipeline.

    Args:
        evals_dir: Path to the evals directory
        exchanges: If set, only evaluate questions for these exchange codes (e.g. ["HKSE"]).
                   Unanswerable questions are always included.

    Returns:
        Dictionary containing evaluation results
    """
    print("Loading golden dataset...")
    golden_path = evals_dir / "golden_dataset.json"
    answerable_qs, unanswerable_qs = load_golden_dataset(golden_path)

    if exchanges:
        answerable_qs = [q for q in answerable_qs if q.get("exchange") in exchanges]
        print(f"Exchange filter: {exchanges} — {len(answerable_qs)} answerable questions selected")

    print(
        f"Loaded {len(answerable_qs)} answerable and {len(unanswerable_qs)} unanswerable questions"
    )

    # Create RAG chain
    print("Creating RAG chain...")
    chain = create_rag_chain()

    # Process all questions
    all_results = []
    correct_refusals = 0

    print("\n=== Processing answerable questions ===")
    for idx, item in enumerate(answerable_qs, 1):
        print(f"[{idx}/{len(answerable_qs)}] {item['question'][:60]}...")
        try:
            result = chain.invoke({"question": item["question"], "exchange_filter": None})

            all_results.append(
                {
                    "question": item["question"],
                    "answer": result["answer"],
                    "contexts": result.get("contexts", []),
                    "ground_truth": item["ground_truth"],
                    "type": "answerable",
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append(
                {
                    "question": item["question"],
                    "answer": f"ERROR: {str(e)}",
                    "contexts": [],
                    "ground_truth": item["ground_truth"],
                    "type": "answerable",
                }
            )

    print("\n=== Processing unanswerable questions ===")
    for idx, item in enumerate(unanswerable_qs, 1):
        print(f"[{idx}/{len(unanswerable_qs)}] {item['question'][:60]}...")
        try:
            result = chain.invoke({"question": item["question"], "exchange_filter": None})

            answer = result["answer"]
            refused = is_refusal(answer)
            if refused:
                correct_refusals += 1
                print("  ✓ Correctly refused")
            else:
                print("  ✗ Did NOT refuse (hallucinated?)")

            all_results.append(
                {
                    "question": item["question"],
                    "answer": answer,
                    "contexts": result.get("contexts", []),
                    "ground_truth": "",
                    "type": "unanswerable",
                    "refused": refused,
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append(
                {
                    "question": item["question"],
                    "answer": f"ERROR: {str(e)}",
                    "contexts": [],
                    "ground_truth": "",
                    "type": "unanswerable",
                    "refused": False,
                }
            )

    # Build HuggingFace Dataset for answerable questions only
    # (RAGAS metrics require ground truth)
    answerable_results = [r for r in all_results if r["type"] == "answerable"]

    # RAGAS 0.2.x renamed fields: question→user_input, answer→response,
    # contexts→retrieved_contexts, ground_truth (list)→reference (str)
    dataset_dict = {
        "user_input": [r["question"] for r in answerable_results],
        "response": [r["answer"] for r in answerable_results],
        "retrieved_contexts": [r["contexts"] for r in answerable_results],
        "reference": [r["ground_truth"] for r in answerable_results],
    }
    dataset = Dataset.from_dict(dataset_dict)

    print(f"\n=== Running RAGAS evaluation on {len(answerable_results)} answerable questions ===")

    # Use Claude as judge LLM and Voyage for embeddings (avoids requiring OpenAI key)
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(
            api_key=get_anthropic_api_key(),
            model="claude-haiku-4-5-20251001",
            temperature=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        VoyageEmbeddings(  # type: ignore[call-arg]
            voyage_api_key=get_voyage_api_key(),
            model="voyage-finance-2",
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Convert to dict for easier handling
    scores = result.to_pandas().to_dict(orient="list")

    # Compute refusal rate
    refusal_rate = correct_refusals / len(unanswerable_qs) if unanswerable_qs else 0.0

    # Aggregate scores
    faithfulness_score = scores["faithfulness"]
    context_precision_score = scores["context_precision"]
    context_recall_score = scores["context_recall"]
    answer_relevancy_score = scores["answer_relevancy"]

    avg_faithfulness = (
        sum(faithfulness_score) / len(faithfulness_score) if faithfulness_score else 0.0
    )
    avg_context_precision = (
        sum(context_precision_score) / len(context_precision_score)
        if context_precision_score
        else 0.0
    )
    avg_context_recall = (
        sum(context_recall_score) / len(context_recall_score) if context_recall_score else 0.0
    )
    avg_answer_relevancy = (
        sum(answer_relevancy_score) / len(answer_relevancy_score) if answer_relevancy_score else 0.0
    )

    # Define thresholds
    thresholds = {
        "faithfulness": 0.85,
        "context_precision": 0.75,
        "refusal_rate": 0.80,
    }

    # Determine pass/fail using thresholds dict
    passed = (
        avg_faithfulness >= thresholds["faithfulness"]
        and avg_context_precision >= thresholds["context_precision"]
        and refusal_rate >= thresholds["refusal_rate"]
    )

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(all_results),
        "answerable_count": len(answerable_qs),
        "unanswerable_count": len(unanswerable_qs),
        "correct_refusals": correct_refusals,
        "refusal_rate": refusal_rate,
        "metrics": {
            "faithfulness": avg_faithfulness,
            "context_precision": avg_context_precision,
            "context_recall": avg_context_recall,
            "answer_relevancy": avg_answer_relevancy,
        },
        "thresholds": thresholds,
        "passed": passed,
        "per_question_scores": scores,
    }

    return results


def print_results(results: dict) -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTimestamp: {results['timestamp']}")
    print(f"Total questions: {results['total_questions']}")
    print(f"  - Answerable: {results['answerable_count']}")
    print(f"  - Unanswerable: {results['unanswerable_count']}")

    print("\nRefusal Handling:")
    print(f"  Correct refusals: {results['correct_refusals']}/{results['unanswerable_count']}")
    print(f"  Refusal rate: {results['refusal_rate']:.2%}")

    print("\nMetrics:")
    metrics = results["metrics"]
    thresholds = results["thresholds"]
    for name, value in metrics.items():
        threshold = thresholds.get(name)
        if threshold is not None:
            status = "✓" if value >= threshold else "✗"
            print(f"  {name}: {value:.4f} (threshold: {threshold}) {status}")
        else:
            print(f"  {name}: {value:.4f}")

    print(f"\nOverall: {'PASSED ✓' if results['passed'] else 'FAILED ✗'}")
    print("=" * 60)


def save_results(results: dict, results_dir: Path) -> Path:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        results_dir: Path to results directory

    Returns:
        Path to saved file
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = results_dir / f"eval_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return output_path


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS Evaluation Harness")
    parser.add_argument(
        "--exchanges",
        nargs="+",
        metavar="EXCHANGE",
        help="Limit eval to these exchange codes, e.g. --exchanges HKSE SGX",
    )
    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent.parent
    evals_dir = project_root / "evals"
    results_dir = evals_dir / "results"

    print("RAGAS Evaluation Harness")
    print("=" * 60)

    # Run evaluation
    results = run_evaluation(evals_dir, exchanges=args.exchanges)

    # Print results
    print_results(results)

    # Save results
    output_path = save_results(results, results_dir)
    print(f"\nResults saved to: {output_path}")

    # Exit with appropriate code
    exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
