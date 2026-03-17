from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """
    Take in a list of document objects that came back from the vector store,
    and return a simple list of strings like URLs or file paths.

    Each document object has a .metadata dictionary inside it.
    That dictionary may or may not have a "source" key.
    If it does, we use that value. If not, we use "Unknown".
    """
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        # (x or {}) if the variable is vaild we return variabl, if it is None or any falsy value, we get an empty dict instead, 
        # getattr safely reads .metadata from the doc object.
        # If the doc doesn't have .metadata at all, we get None,
        # and "or {}" turns that into an empty dictionary so
        # the .get("source") call below won't crash.
        # := is the "walrus operator" that lets us assign this metadata to a variable
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]
    
    """
        # Without walrus operator :=, we need a longer version
        sources = []
        for doc in context_docs:
            meta = getattr(doc, "metadata", None) or {}    # = works here because it's its own line
            if meta is not None:
                sources.append(str(meta.get("source") or "Unknown"))
    """


# ---------- Page Setup ----------
# This must be the very first Streamlit command in the file.
# It controls what appears in the browser tab and how wide the page is.
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

# ---------- Sidebar ----------
# The sidebar sits on the left side of the page.
# with st.sidebar: means "put everything inside this block into the sidebar"
# We put a "Clear chat" button here so the user can wipe the
# conversation and start fresh at any time.
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        # Remove the saved messages from session_state.
        # use_container_width=True makes the button stretch to fill the sidebar, which looks nicer.
        # session_state is like a storage box that keeps data alive
        # even when Streamlit re-runs the whole script (which it does
        # on every user interaction).
        st.session_state.pop("messages", None)
        # seesion_state is a dictionary-like object, so we use .pop() to remove the "messages" key and its value.
        # Force the page to re-run right now so the user sees
        # the empty chat immediately instead of waiting for
        # their next action.
        st.rerun()

# ---------- Create Chat History on First Visit ----------
# When the user opens the app for the first time (or after clearing),
# there is no "messages" key in session_state yet.
# We create it here with one welcome message so the chat isn't blank.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I'll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

# ---------- Display All Past Messages ----------
# Every time the user does something (sends a message, clicks a button),
# Streamlit re-runs this entire script from top to bottom.
# So we need to loop through ALL saved messages and draw them again
# each time, otherwise old messages would disappear.
for msg in st.session_state.messages:
    # chat_message creates a bubble on the left (assistant) or right (user)
    # chat_message is a context manager
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Some messages have source links attached to them.
        # We show these inside a collapsible section so they
        # don't take up too much space on screen.
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# ---------- Handle New User Input ----------
# chat_input puts a text box at the bottom of the page.
# It returns the text the user typed when they press Enter.
# If they haven't typed anything yet, it returns None.
prompt = st.chat_input("Ask a question about LangChain…")

if prompt:
    # Step 1: Save the user's message into session_state so it
    # won't be lost on the next re-run, then show it on screen.
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 2: Generate the assistant's answer
    with st.chat_message("assistant"):
        try:
            # Show a spinning animation while we wait for the answer.
            # The RAG pipeline can take a few seconds because it needs to
            # search a database and call the LLM.
            with st.spinner("Retrieving docs and generating answer…"):
                # run_llm does three things:
                #   1. Turns the user's question into a number vector (embedding)
                #   2. Searches the vector store to find the most relevant
                #      chunks of LangChain documentation
                #   3. Sends those chunks along with the question to the LLM,
                #      which writes an answer based on that context
                #
                # It returns a dictionary with two keys:
                #   "answer"  -> the text answer from the LLM
                #   "context" -> the list of document chunks it found
                result: Dict[str, Any] = run_llm(prompt)

                # Pull the answer out of the result.
                # If the answer is missing or blank, show a fallback message
                # so the user isn't left staring at an empty bubble.
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"

                # Pull out the source URLs/paths from the retrieved documents
                # so we can show the user where the information came from.
                sources = _format_sources(result.get("context", []))

            # The spinner disappears once we reach this point.
            # Now we display the actual answer.
            st.markdown(answer)

            # If we found any sources, show them in a collapsible section.
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # Step 3: Save the assistant's response to session_state.
            # This is important — without this, the answer would vanish
            # the next time the user sends a new message, because Streamlit
            # re-runs the script and only draws what's in session_state.
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            # If anything goes wrong (network issue, bad API key, etc.),
            # show an error message instead of crashing the whole app.
            # We do NOT save this to session_state, so failed attempts
            # won't appear in the chat history on the next re-run.
            st.error("Failed to generate a response.")
            st.exception(e)