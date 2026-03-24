"""Streamlit chat UI for the vulne-explorer FastAPI backend."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import streamlit as st


API_BASE_URL = os.getenv("VULN_EXPLORER_API_URL", "http://localhost:8000")


async def _send_chat_message(message: str) -> dict[str, Any]:
    """Send a chat request to the FastAPI backend asynchronously.

    Input:
        A user message string for vulnerability analysis.
    Output:
        Returns the parsed JSON response from the `/chat` endpoint.
    Security context:
        Sends only the user message to the backend API and keeps network access
        scoped to the configured application endpoint.
    """

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0) as client:
        response = await client.post("/chat", json={"message": message})
        response.raise_for_status()
        return response.json()


def _extract_cve_ids(retrieved_documents: list[dict[str, Any]]) -> list[str]:
    """Extract source CVE identifiers from retrieved document metadata.

    Input:
        The `retrieved_documents` array returned by the FastAPI backend.
    Output:
        Returns a de-duplicated list of CVE IDs.
    Security context:
        Uses only explicit metadata returned by the backend, avoiding any
        hidden client-side inference about vulnerability provenance.
    """

    cve_ids: list[str] = []
    for document in retrieved_documents:
        metadata = document.get("metadata", {})
        cve_id = metadata.get("cve_id")
        if cve_id and cve_id not in cve_ids:
            cve_ids.append(str(cve_id))
    return cve_ids


def _render_remediation_plan(answer: str) -> None:
    """Render the remediation plan in a markdown-friendly format.

    Input:
        The final generation text returned by the backend.
    Output:
        Writes the answer to the Streamlit app.
    Security context:
        Displays only backend-generated content and does not execute or render
        arbitrary HTML from the model output.
    """

    st.markdown("### Remediation Plan")
    st.markdown(answer or "Information not found in database.")


def main() -> None:
    """Run the Streamlit chat interface for vulne-explorer.

    Input:
        Uses Streamlit session state and the configured API base URL.
    Output:
        Renders a chat interface connected to the FastAPI backend.
    Security context:
        Keeps user interaction limited to backend-mediated inference so all
        vulnerability reasoning remains server-side and auditable.
    """

    st.set_page_config(page_title="vulne-explorer", page_icon=":shield:", layout="wide")
    st.title("vulnerability explorer")
    st.caption("Real-time vulnerability intelligence and remediation guidance.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant":
                _render_remediation_plan(entry["content"])
                source_cve_ids = entry.get("source_cve_ids", [])
                with st.expander("Source CVE IDs", expanded=False):
                    if source_cve_ids:
                        for cve_id in source_cve_ids:
                            st.markdown(f"- `{cve_id}`")
                    else:
                        st.markdown("No source CVE IDs were returned.")
            else:
                st.markdown(entry["content"])

    prompt = st.chat_input("Ask about a CVE, package, exploit, or remediation plan")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing vulnerabilities..."):
            try:
                result = asyncio.run(_send_chat_message(prompt))
                answer = result.get("final_generation", "Information not found in database.")
                retrieved_documents = result.get("retrieved_documents", [])
                source_cve_ids = _extract_cve_ids(retrieved_documents)
                _render_remediation_plan(answer)
                with st.expander("Source CVE IDs", expanded=False):
                    if source_cve_ids:
                        for cve_id in source_cve_ids:
                            st.markdown(f"- `{cve_id}`")
                    else:
                        st.markdown("No source CVE IDs were returned.")
            except httpx.HTTPError as exc:
                answer = "Backend request failed. Check API availability and configuration."
                source_cve_ids = []
                st.error(f"{answer}\n\nDetails: {exc}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "source_cve_ids": source_cve_ids,
        }
    )


if __name__ == "__main__":
    main()
