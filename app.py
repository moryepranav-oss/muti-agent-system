import importlib

try:
    st = importlib.import_module("streamlit")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "streamlit is not installed. Please run: pip install streamlit"
    ) from exc

import re
import time

from tools import web_search, scrape_url
from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

try:
    from agents import agent_llm
    AGENT_MODEL = agent_llm.model_name
except ImportError:
    AGENT_MODEL = "OLD agents.py (agent_llm missing)"

# Groq free tier has token limits, so we cap how much text goes to the writer.
MAX_SEARCH_CHARS = 1500
MAX_SCRAPED_CHARS = 2000

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔎 Research Agent")
    st.caption("Multi-agent pipeline")
    st.caption(f"Tool agents model: {AGENT_MODEL}")
    st.caption("app.py version: 3 (tool-call fallback on)")
    st.markdown(
        """
**Pipeline steps**
1. 🔍 Search agent
2. 📄 Reader agent (scraper)
3. ✍️ Writer chain
4. 🧐 Critic chain
"""
    )
    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("state", None)
        st.session_state.pop("topic", None)
        st.rerun()

# ---------- Main ----------
st.title("Multi-Agent Research Assistant")
st.write("Enter a topic and the agents will search, read, write and review a report for you.")

topic = st.text_input("Research topic", placeholder="e.g. Latest advances in quantum computing")
run = st.button("🚀 Run research", type="primary", disabled=not topic.strip())


def as_text(value) -> str:
    """Agents/chains may return a string, a message object, or a list of blocks."""
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(value)


def invoke_with_retry(fn, payload, retries=3, wait=25):
    """Call fn(payload); if Groq says rate limit (429), wait and try again."""
    for attempt in range(retries):
        try:
            return fn(payload)
        except Exception as e:
            is_limit = "429" in str(e) or "rate limit" in str(e).lower()
            if is_limit and attempt < retries - 1:
                st.write(f"⏳ Rate limit reached. Waiting {wait} seconds, then retrying...")
                time.sleep(wait)
            else:
                raise


def is_tool_error(e: Exception) -> bool:
    """True when the model made a bad tool call (Groq returns a 400 tool_use_failed)."""
    msg = str(e)
    return "tool_use_failed" in msg or "Tool call validation failed" in msg


def search_step(topic: str) -> str:
    try:
        agent = build_search_agent()
        result = invoke_with_retry(agent.invoke, {
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        return as_text(result["messages"][-1].content)
    except Exception as e:
        if is_tool_error(e):
            st.write("⚠️ Agent made a bad tool call, using direct search instead...")
            return as_text(web_search.invoke({"query": topic}))
        raise


def reader_step(topic: str, search_results: str) -> str:
    try:
        agent = build_reader_agent()
        result = invoke_with_retry(agent.invoke, {
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{search_results[:800]}"
            )]
        })
        return as_text(result["messages"][-1].content)
    except Exception as e:
        if is_tool_error(e):
            st.write("⚠️ Agent made a bad tool call, scraping the first URL directly...")
            urls = re.findall(r"https?://[^\s)\]\"'>]+", search_results)
            if urls:
                return as_text(scrape_url.invoke({"url": urls[0]}))
            return "No URL found to scrape."
        raise


def run_pipeline(topic: str) -> dict:
    """Same logic as pipeline.py, but with live status updates for the UI."""
    state = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # Step 1 - Search
        st.write("🔍 **Step 1:** Search agent is working...")
        state["search_results"] = search_step(topic)
        st.write("✅ Search complete")

        # Step 2 - Reader
        st.write("📄 **Step 2:** Reader agent is scraping top resources...")
        state["scraped_content"] = reader_step(topic, state["search_results"])
        st.write("✅ Scraping complete")

        # Step 3 - Writer
        st.write("✍️ **Step 3:** Writer is drafting the report...")
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results'][:MAX_SEARCH_CHARS]} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content'][:MAX_SCRAPED_CHARS]}"
        )
        state["report"] = as_text(invoke_with_retry(writer_chain.invoke, {
            "topic": topic,
            "research": research_combined,
        }))
        st.write("✅ Report drafted")

        # Step 4 - Critic
        st.write("🧐 **Step 4:** Critic is reviewing the report...")
        state["feedback"] = as_text(invoke_with_retry(critic_chain.invoke, {"report": state["report"]}))
        st.write("✅ Review complete")

        status.update(label="Pipeline finished!", state="complete", expanded=False)

    return state


if run:
    try:
        st.session_state["state"] = run_pipeline(topic.strip())
        st.session_state["topic"] = topic.strip()
    except Exception as e:
        msg = str(e)
        if "rate" in msg.lower() or "429" in msg:
            st.error("Rate limit reached. Please wait a minute and try again.")
        else:
            st.error(f"Something went wrong: {msg}")

# ---------- Results ----------
state = st.session_state.get("state")
if state:
    st.divider()
    st.subheader(f"Results: {st.session_state.get('topic', '')}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📝 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📄 Scraped Content"]
    )

    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(state["feedback"])

    with tab_search:
        st.markdown(state["search_results"])

    with tab_scraped:
        st.markdown(state["scraped_content"])