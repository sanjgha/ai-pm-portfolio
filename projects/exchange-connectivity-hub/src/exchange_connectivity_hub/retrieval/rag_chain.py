"""RAG chain: retrieve → rerank → prompt → LLM → output parser."""

from datetime import datetime, timezone
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic

from exchange_connectivity_hub.config import get_config, get_anthropic_api_key
from exchange_connectivity_hub.retrieval.retriever import create_retriever
from exchange_connectivity_hub.retrieval.reranker import rerank_documents


def _format_docs(docs: list[Any]) -> str:
    """Format documents for prompt."""
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        content = doc.page_content
        context_parts.append(f"[Source: {source}, page {page}]\n{content}")
    return "\n\n".join(context_parts)


def _extract_sources(docs: list[Any]) -> list[dict[str, Any]]:
    """Extract source metadata from documents."""
    sources = []
    seen = set()  # Dedupe by filename + page

    for doc in docs:
        filename = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number")
        exchange = doc.metadata.get("exchange", "Unknown")
        ingested_at = doc.metadata.get("ingested_at", "Unknown")

        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "filename": filename,
                    "page_number": page,
                    "exchange": exchange,
                    "ingested_at": ingested_at,
                }
            )

    return sources


def _check_staleness(sources: list[dict[str, Any]]) -> str | None:
    """Check if any source is older than staleness threshold."""
    config = get_config()
    staleness_days = config["staleness"]["days"]

    for source in sources:
        ingested_at = source.get("ingested_at")
        if ingested_at and ingested_at != "Unknown":
            try:
                ingest_date = datetime.fromisoformat(ingested_at)
                days_ago = (datetime.now(timezone.utc) - ingest_date).days

                if days_ago > staleness_days:
                    return f"⚠️ Some sources are over {staleness_days} days old (last re-verified {days_ago} days ago). Verify against exchange website."
            except (ValueError, TypeError):
                continue

    return None


def create_rag_chain() -> Any:
    """Create the RAG chain.

    Chain flow:
    1. Retrieve docs (via retriever)
    2. Rerank docs
    3. Format context
    4. Generate answer (via Claude)
    5. Extract sources and check staleness

    Returns:
        Runnable chain that accepts {"question": str, "exchange_filter": str | None}
    """
    config = get_config()

    # Prompt template
    prompt_template = """Answer the following question using only the context provided below.

Context:
{context}

Question: {question}

Instructions:
- Answer from the context only
- Cite the source filename and page number for each fact
- If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
- Never hallucinate or make up information

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # LLM
    llm = ChatAnthropic(
        api_key=get_anthropic_api_key(),
        model=config["models"]["llm"],
        temperature=0,
    )

    def _retrieve_and_rerank(inputs: dict[str, Any]) -> dict[str, Any]:
        """Retrieve and rerank documents."""
        question = inputs["question"]
        exchange_filter = inputs.get("exchange_filter")

        # Create retriever
        retriever = create_retriever(
            exchange_filter=exchange_filter,
        )

        # Retrieve docs
        docs = retriever.invoke(question)

        # Rerank
        config = get_config()
        reranked = rerank_documents(
            docs,
            query=question,
            top_n=config["retrieval"]["rerank_top_n"],
        )

        return {"docs": reranked, "question": question}

    def _generate_response(inputs: dict[str, Any]) -> dict[str, Any]:
        """Generate final response with sources and staleness check."""
        docs = inputs["docs"]
        question = inputs["question"]

        # Format context
        context = _format_docs(docs)

        # Generate answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # Extract sources
        sources = _extract_sources(docs)

        # Check staleness
        staleness_warning = _check_staleness(sources)

        return {
            "answer": answer,
            "sources": sources,
            "staleness_warning": staleness_warning,
        }

    # Full chain
    chain = RunnablePassthrough.assign() | _retrieve_and_rerank | _generate_response

    return chain
