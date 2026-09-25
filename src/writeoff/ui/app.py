"""WriteOff Assistant web UI (Streamlit). Run: streamlit run src/writeoff/ui/app.py

Talks to the API only (WRITEOFF_API_URL, API_TOKEN). Answers stream progress while the
agent researches, then show the verified answer with a collapsible Sources panel.
"""

import os
from typing import Any

import streamlit as st

from writeoff.ui.client import APIError, WriteOffClient

ENTITY_TYPES = {
    "Not specified": None,
    "Sole proprietor / single-member LLC": "sole_prop",
    "Partnership / multi-member LLC": "partnership",
    "S corporation": "s_corp",
    "C corporation": "c_corp",
}
TAX_YEARS = [int(y) for y in os.environ.get("WRITEOFF_TAX_YEARS", "2025,2026").split(",")]
TREATMENT_LABELS = {
    "fully_deductible": "Deductible",
    "partially_deductible": "Partly deductible",
    "capitalize_and_depreciate": "Capitalize / depreciate",
    "not_deductible": "Not deductible",
    "depends_on_facts": "Depends on facts",
}


@st.cache_resource
def client() -> WriteOffClient:
    return WriteOffClient.from_env()


def ensure_session(entity_type: str | None, tax_year: int) -> str:
    """Create the API session on first use; keep it in sync with the sidebar."""
    state = st.session_state
    wanted = (entity_type, tax_year)
    if "session_id" not in state:
        state.session_id = client().create_session(entity_type, tax_year)["session_id"]
    elif state.get("session_facts") != wanted:
        client().update_session(state.session_id, entity_type, tax_year)
    state.session_facts = wanted
    return str(state.session_id)


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            title = source.get("title") or ""
            url = source.get("url")
            heading = f"**{source['citation']}**"
            if title:
                heading += f" · [{title}]({url})" if url else f" · {title}"
            st.markdown(heading)
            st.text(source["text"])


def render_verification(verification: dict[str, Any] | None) -> None:
    if not verification or verification["status"] == "skipped":
        return
    narrowed = verification["partially_supported"] + verification["unsupported"]
    note = f"Checked against the sources: {verification['supported']} claim(s) supported"
    if narrowed:
        note += f", {narrowed} narrowed or removed"
    if verification["status"] == "error":
        note = "Automatic source checking was unavailable for this answer."
    st.caption(note)


def render_answer(message: dict[str, Any]) -> None:
    st.markdown(message["content"])
    render_verification(message.get("verification"))
    render_sources(message.get("sources", []))
    unconfirmed = (message.get("verification") or {}).get("unconfirmed") or []
    if unconfirmed:
        with st.expander("What couldn't be confirmed"):
            for item in unconfirmed:
                st.markdown(f"- {item}")


def ask_tab(entity_type: str | None, tax_year: int) -> None:
    for message in st.session_state.setdefault("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message)
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask about a business expense…")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            session_id = ensure_session(entity_type, tax_year)
            answer: dict[str, Any] | None = None
            with st.status("Researching…", expanded=False) as status:
                for event in client().stream_chat(question, session_id):
                    if event.event == "progress":
                        status.update(label=event.data["message"])
                        status.write(event.data["message"])
                    elif event.event == "answer":
                        answer = event.data
                    elif event.event == "error":
                        raise APIError(500, event.data.get("message"))
                status.update(label="Done", state="complete")
        except APIError as exc:
            st.error(f"Sorry, that didn't work: {exc}")
            return
        if answer is None:
            st.error("The connection closed before the answer arrived. Please try again.")
            return
        message = {
            "role": "assistant",
            "content": answer["answer"],
            "sources": answer["sources"],
            "verification": answer["verification"],
        }
        render_answer(message)
        st.session_state.messages.append(message)


def upload_tab(entity_type: str | None, tax_year: int) -> None:
    st.markdown(
        "Upload a CSV with a **description** and an **amount** column (optionally "
        "**business_use_pct**). Rows get a first-pass classification from fixed rules; "
        "ask about any row in the chat for a sourced answer. Cell contents are treated "
        "as data only, and files aren't stored."
    )
    if entity_type is None:
        st.info("Choose your entity type in the sidebar first.")
        return
    upload = st.file_uploader("Expense CSV", type=["csv"])
    if upload is None or not st.button("Classify expenses"):
        return
    try:
        result = client().upload_csv(upload.name, upload.getvalue(), entity_type, tax_year)
    except APIError as exc:
        st.error(str(exc))
        return
    columns = st.columns(3)
    columns[0].metric("Total spent", f"${float(result['total_amount']):,.2f}")
    columns[1].metric("Deductible (first pass)", f"${float(result['total_deductible']):,.2f}")
    columns[2].metric("Rows needing review", result["rows_needing_review"])
    st.dataframe(
        [
            {
                "Row": r["line"],
                "Description": r["description"],
                "Amount": float(r["amount"]),
                "Business %": float(r["business_use_pct"]),
                "Category": r["category"],
                "Treatment": TREATMENT_LABELS.get(r["treatment"], r["treatment"]),
                "Deductible": None
                if r["deductible_amount"] is None
                else float(r["deductible_amount"]),
                "Authorities": ", ".join(r["authorities"]),
                "Open questions": " ".join(r["questions"]),
            }
            for r in result["rows"]
        ],
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="WriteOff Assistant", layout="wide")
    with st.sidebar:
        st.title("WriteOff Assistant")
        st.caption("Federal income-tax deductions for U.S. small businesses")
        entity_label = st.selectbox("Entity type", list(ENTITY_TYPES))
        tax_year = st.selectbox("Tax year", TAX_YEARS)
        if st.button("New conversation"):
            for key in ("session_id", "session_facts", "messages"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.caption(
            "General information, not tax or legal advice. Consult a CPA or enrolled agent "
            "about your situation."
        )
    entity_type = ENTITY_TYPES[entity_label]
    ask, upload = st.tabs(["Ask", "Upload expenses"])
    with ask:
        ask_tab(entity_type, tax_year)
    with upload:
        upload_tab(entity_type, tax_year)


main()
