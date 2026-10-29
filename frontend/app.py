"""DrugAssist presentation only: every account and search action goes through FastAPI."""

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
st.set_page_config(page_title="DrugAssist", page_icon="💊", layout="centered")
API_URL = os.getenv("DRUGASSIST_API_URL", "http://127.0.0.1:8000").rstrip("/")


def clear_session(message: str) -> None:
    """Clear browser session state, save a notice, and rerun the interface."""
    st.session_state.clear()
    st.session_state["notice"] = message
    st.rerun()


def api_request(
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    on_error: Callable[[], None] | None = None,
) -> dict:
    """Call the backend with session credentials and display sanitized errors."""

    def fail(message: str) -> None:
        """Run the error callback, display a message, and stop the current render."""
        if on_error is not None:
            on_error()
        st.error(message)
        st.stop()

    headers = {}
    if "token" in st.session_state:
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    try:
        timeout = httpx.Timeout(float(os.getenv("DRUGASSIST_REQUEST_TIMEOUT", "120")), connect=5)
        response = httpx.request(
            method,
            API_URL + path,
            json=payload,
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        )
    except httpx.RequestError:
        fail("Cannot reach DrugAssist. Check that the backend is running and try again.")
    except ValueError:
        fail("Check the request timeout in your application settings.")
    if response.status_code == 401:
        if "token" in st.session_state:
            clear_session("Your session has ended. Please log in again.")
        fail("Incorrect email or password.")
    messages = {
        409: "An account with this email already exists. Choose Log in.",
        422: "Please check the information you entered. Passwords must contain 8–128 characters.",
        429: "Too many attempts. Please wait before trying again.",
        503: "The service is temporarily unavailable. Please try again later.",
    }
    if not 200 <= response.status_code < 300:
        fail(messages.get(response.status_code, "Could not complete the request. Please try again."))
    if response.status_code == 204:
        return {}
    try:
        return response.json()
    except ValueError:
        fail("The backend returned an unexpected response. Please try again.")


def account_screen() -> None:
    """Render account forms and submit signup or login requests."""
    screen = st.radio("Account", ["Log in", "Sign up"], horizontal=True, label_visibility="collapsed")
    with st.form("account", clear_on_submit=True):
        st.subheader(screen)
        email = st.text_input("Email", max_chars=254)
        password = st.text_input("Password", type="password", max_chars=128)
        confirmation = (
            st.text_input("Confirm password", type="password", max_chars=128) if screen == "Sign up" else None
        )
        st.caption("Use a password with at least 8 characters.")
        submitted = st.form_submit_button(screen, type="primary", width="stretch")
    if submitted:
        if not email.strip() or not 8 <= len(password) <= 128:
            st.error("Enter an email and a password with 8–128 characters.")
        elif confirmation is not None and password != confirmation:
            st.error("The passwords do not match.")
        else:
            with st.spinner("Please wait…"):
                result = api_request(
                    "POST",
                    "/auth/signup" if screen == "Sign up" else "/auth/login",
                    {"email": email.strip(), "password": password},
                )
            if screen == "Sign up":
                st.success("Account created. Choose Log in to continue.")
            else:
                st.session_state["token"] = result["session_token"]
                st.rerun()


def label_date(value: str | None) -> str:
    """Format a valid FDA label date or return a missing-date placeholder."""
    if value and len(value) == 8 and value.isdigit():
        try:
            return datetime.strptime(value, "%Y%m%d").strftime("%d %b %Y")
        except ValueError:
            pass
    return "Not provided"


def show_answer(result: dict) -> None:
    """Display generated statements and their supporting FDA source windows."""
    st.subheader("Answer")
    status = result.get("answer_status")
    if status != "generated":
        st.info(result["answer_note"])
        if result.get("evidence_note"):
            st.text(result["evidence_note"])
        return
    st.caption("AI-written from the FDA evidence below. Review the sources for important qualifications.")
    current_section = "answer"
    headings = {"details": "Details", "warnings": "Important warnings"}
    for statement in result["statements"]:
        section = statement.get("section", "answer")
        if section != current_section and section in headings:
            st.markdown(f"**{headings[section]}**")
            current_section = section
        # Model text is displayed literally, so it cannot introduce clickable invented URLs or HTML.
        st.text(statement["text"])
        st.caption("Sources: " + ", ".join(dict.fromkeys(statement["source_ids"])))
    partial = any(source["context_partial"] for source in result["citations"])
    if result.get("evidence_note") or partial:
        st.markdown("**What this answer covers**")
        if result.get("evidence_note"):
            st.text(result["evidence_note"])
        if partial:
            st.caption(
                "The supporting excerpts include part of a longer section; this may not be a "
                "complete answer."
            )
    with st.expander("Supporting FDA context"):
        for source in result["citations"]:
            st.markdown(f"**{source['source_id']} | {source['section'].replace('_', ' ').title()}**")
            st.text(source["drug_name"])
            st.caption(f"Label date: {label_date(source.get('effective_time'))}")
            if source.get("source_url") and urlsplit(source["source_url"]).scheme == "https":
                st.link_button(f"FDA source {source['source_id']}", source["source_url"])
            if source["context_partial"]:
                st.caption("Selected context from a longer section; this may not be a complete list.")
            st.text(source["text"])
    st.caption(f"Wording: {result['audience'].title()} | {result['answer_note']}")


def show_results(result: dict, elapsed: float | None = None) -> None:
    """Display ranked passages, no-match guidance, and source metadata."""
    if "answer_status" in result:
        show_answer(result)
    st.subheader("Results")
    hits = result["results"]
    labels = len({hit["set_id"] for hit in hits})
    summary = (
        f"{len(hits)} {'passage' if len(hits) == 1 else 'passages'} "
        f"from {labels} FDA label {'set' if labels == 1 else 'sets'}"
    )
    if elapsed is not None:
        summary += f" | Search time: {elapsed:.1f}s"
    st.caption(summary)
    st.text(f"Question: {result['query']}")
    if result["no_match"]:
        st.info("No good match found.")
        hints = {
            "outside_supported_label_scope": (
                "Try a drug name in this collection and a label topic such as uses, dosage or "
                "warnings."
            ),
            "below_experimental_threshold": (
                "No passage met the current relevance cutoff. Try a more specific drug or "
                "label section."
            ),
            "no_candidates": "No passage matched this drug and section. Try a broader question.",
        }
        st.caption(
            hints.get(
                result.get("no_match_reason"), "Try another drug name or ask about a specific label section."
            )
        )
        return
    parsed = result.get("parsed", {})
    context = [
        str(value).replace("_", " ")
        for value in (parsed.get("drug"), parsed.get("information_type"))
        if value
    ]
    if context:
        st.caption("Search focus: " + " | ".join(context))
    st.caption("Ordered by relevance. Scores are not probabilities of correctness.")
    for hit in hits:
        with st.container(border=True):
            heading, source = st.columns([3, 1], vertical_alignment="center")
            heading.subheader(f"{hit['rank']}. {hit['drug_name']}")
            if hit.get("source_url") and urlsplit(hit["source_url"]).scheme == "https":
                source.link_button("FDA source", hit["source_url"], width="stretch")
            section = hit["section"].replace("_", " ").title()
            st.caption(f"{section} | Label date: {label_date(hit.get('effective_time'))}")
            st.caption(f"{result['score_kind']}: {hit['score']:.3f}")
            with st.expander("Read FDA passage", expanded=hit["rank"] == 1):
                st.text(hit["text"])
    with st.expander("Search and source details"):
        st.caption(result["score_note"])
        st.json(
            {
                "method": result["method"],
                "snapshot_id": result["snapshot_id"],
                "applied_filters": result["applied_filters"],
                "threshold": result["threshold"],
                "passages": [
                    {
                        key: hit.get(key)
                        for key in (
                            "rank",
                            "label_id",
                            "set_id",
                            "document_id",
                            "chunk_id",
                            "effective_time",
                            "source_record_hash",
                            "source_body_sha256",
                        )
                    }
                    for hit in hits
                ],
            }
        )


def search_screen(user: dict) -> None:
    """Render the search form and manage the signed-in user's saved results."""
    identity, logout = st.columns([4, 1], vertical_alignment="center")
    identity.caption(f"Signed in as {user['email']}")
    if logout.button("Log out", width="stretch"):
        api_request("POST", "/auth/logout")
        clear_session("You have been logged out.")
    with st.form("search"):
        question = st.text_area(
            "Your question",
            placeholder="What are the contraindications of metformin?",
            max_chars=2000,
            height=110,
        )
        writing, audience_column = st.columns([2, 1])
        write_answer = writing.checkbox("Write a cited answer", value=True)
        audience = audience_column.selectbox("Wording", ["Patient", "Clinician"])
        count, action = st.columns([1, 2], vertical_alignment="bottom")
        k = count.number_input("Number of results", min_value=1, max_value=20, value=5, step=1)
        with action:
            submitted = st.form_submit_button("Search FDA labels", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("results", None)
        st.session_state.pop("search_seconds", None)
        if not question.strip():
            st.error("Enter a drug-label question to search.")
        else:
            progress = st.empty()
            with progress.container(), st.status("Searching FDA labels", expanded=True) as status:
                st.caption(
                    "Finding and ranking FDA passages, then writing a cited answer."
                    if write_answer else "Finding matching passages and ranking their relevance."
                )
                started = perf_counter()
                st.session_state["results"] = api_request(
                    "POST",
                    "/answer" if write_answer else "/search",
                    {
                        "query": question.strip(),
                        "k": int(k),
                        **({"audience": audience.lower()} if write_answer else {}),
                    },
                    on_error=lambda: status.update(
                        label="Search not completed", state="error", expanded=True
                    ),
                )
                st.session_state["search_seconds"] = perf_counter() - started
            progress.empty()
    result = st.session_state.get("results")
    if result:
        show_results(result, st.session_state.get("search_seconds"))
    st.divider()
    st.caption(
        result["disclaimer"]
        if result
        else (
            "FDA label excerpts are for information only, not medical advice. Consult a "
            "qualified healthcare professional."
        )
    )


def main() -> None:
    """Render the account or search screen according to session authentication."""
    url = urlsplit(API_URL)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        st.error("Set a local API URL, or use HTTPS for a remote backend.")
        st.stop()
    st.title("DrugAssist")
    st.caption("Find information in FDA drug labels.")
    if notice := st.session_state.pop("notice", None):
        st.info(notice)
    if "token" in st.session_state:
        user = api_request("GET", "/auth/me")
        search_screen(user)
    else:
        account_screen()


if __name__ == "__main__":
    main()
