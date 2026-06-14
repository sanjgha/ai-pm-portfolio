"""Streamlit demo: type a rule -> see JSON + plain-English read-back -> approve."""

import streamlit as st

from pre_trade_risk_rules_assistant import store
from pre_trade_risk_rules_assistant.graph import run_graph

st.set_page_config(page_title="RuleForge", page_icon="⚙️")
st.title("⚙️ RuleForge — Pre-Trade Risk Rules Assistant")
st.caption("Natural-language risk rule → validated, audit-logged JSON config.")

example = "Block any single buy order over 5 million SGD on SGX small-caps"
request = st.text_area("Describe the rule", value=example, height=100)

if st.button("Draft rule", type="primary"):
    with st.spinner("Running generate → validate → self-correct…"):
        state = run_graph(request)

    if state.get("status") == "ok":
        st.success(f"Validated after {state.get('attempts', 0)} self-correction(s).")
        st.subheader("Plain-English read-back")
        st.info(state.get("readback", ""))
        st.subheader("Generated config")
        st.json(state["validated_rule"])
        if st.button("✅ Approve & persist"):
            rule_id = store.save_rule(state["validated_rule"])
            st.success(f"Saved rule {rule_id} (audit-logged).")
    else:
        st.error("Escalated to human — could not produce a valid rule.")
        st.write("Diagnostic:")
        for err in state.get("errors", []):
            st.write(f"- {err}")
