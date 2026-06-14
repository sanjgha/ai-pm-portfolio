"""Streamlit UI for exchange connectivity Q&A."""

import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="Exchange Connectivity Hub",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Exchange Connectivity Hub")
st.markdown("Ask questions about Asian exchange (SGX, HKSE, TSE) connectivity and trading rules.")

# API URL (default to localhost)
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="FastAPI backend URL",
)

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.success(f"✅ API Connected ({health.get('collection_count', 'N/A')} docs)")
except Exception:
    st.sidebar.error("❌ API Disconnected")

# Question input
question = st.text_input(
    "Question",
    placeholder="e.g., What is the minimum lot size for SGX equities?",
    help="Ask about trading rules, order types, lot sizes, etc.",
)

# Exchange filter
exchange_filter = st.selectbox(
    "Filter by Exchange",
    options=["All", "SGX", "HKSE", "TSE"],
    index=0,
)

# Query button
if st.button("Ask", type="primary"):
    if not question or not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Prepare request
    payload = {
        "question": question,
        "exchange_filter": exchange_filter if exchange_filter != "All" else None,
    }

    # Call API
    with st.spinner("Retrieving answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Display staleness warning
            if result.get("staleness_warning"):
                st.warning(result["staleness_warning"])

            # Display answer
            st.markdown("### Answer")
            st.markdown(result["answer"])

            # Display sources
            if result.get("sources"):
                st.markdown("### Sources")
                for source in result["sources"]:
                    filename = source.get("filename", "Unknown")
                    page = source.get("page_number", "?")
                    exchange = source.get("exchange", "Unknown")
                    ingested_at = source.get("ingested_at", "Unknown")

                    st.markdown(
                        f"- **{filename}** (page {page}, {exchange}) – ingested {ingested_at}"
                    )

        except requests.exceptions.RequestException as e:
            st.error(f"Error querying API: {e}")

# Example questions
st.markdown("---")
st.markdown("### Example Questions")
examples = [
    ("What is the minimum lot size for SGX equities?", "SGX"),
    ("Does HKSE support iceberg orders?", "HKSE"),
    ("What are TSE trading hours?", "TSE"),
]

for q, ex in examples:
    if st.button(f"{q}", key=q):
        st.session_state["question"] = q
        st.session_state["exchange_filter"] = ex
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Voyage AI, Claude, LangChain, and RAGAS*")
