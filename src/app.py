import streamlit as st
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import hashlib

# --- Setup connections ---
client = MilvusClient(
    uri="http://milvus_db:19530"
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Ollama(
    model="llama3",
    request_timeout=300.0,
    host="http://ollama_llm:11434",
)

COURSE_COLLECTIONS = {
    "F21CA": "HWU_MACS_F21CA",
    "F21NL": "HWU_MACS_F21NL",
    # Add more courses here
}

# --- Simple cache helpers ---
def get_cache_key(user_query: str, chat_history: list, course_id: str) -> str:
    """Hash user query + recent history + course_id into a cache key."""
    history_text = "".join([f"{m['role']}{m['content']}" for m in chat_history[-4:]])
    return hashlib.sha256((user_query + history_text).encode()).hexdigest()

def get_from_cache(cache_name: str, key: str):
    return st.session_state[cache_name].get(key, None)

def set_cache(cache_name: str, key: str, value):
    st.session_state[cache_name][key] = value

def rewrite_query(user_query: str, chat_history: list, llm) -> str:
    """
    Rewrite the user's query into a standalone, context-rich question 
    using recent chat history.
    """
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat_history[-4:]]  # last 2 user+assistant turns
    )
    prompt = (
        "You are a query rewriter. The user may ask follow-up questions. "
        "Rewrite the latest user query into a fully self-contained question "
        "that can be understood without conversation history.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User query: {user_query}\n"
        "Rewritten query:"
    )

    # Just a single-turn call to the LLM
    rewrite_response = llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
    return rewrite_response.message.content.strip()

# --- Streamlit UI ---

# Initialise session state for app view
if "current_view" not in st.session_state:
    st.session_state.current_view = "selection"
if "selected_course_id" not in st.session_state:
    st.session_state.selected_course_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rewrite_cache" not in st.session_state:
    st.session_state.rewrite_cache = {}
if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}

#  --- Selection Page ---
if st.session_state.current_view == "selection":
    st.title("HWU MACS Learning Buddy")
    st.subheader("Select a course to get started:")

    selected_course = st.selectbox("Choose a course", list(COURSE_COLLECTIONS.keys()))

    if st.button("Start Chat"):
        st.session_state.selected_course_id = selected_course
        st.session_state.messages = []
        st.session_state.current_view = "chat"
        st.rerun()

# --- Chat Page ---
elif st.session_state.current_view == "chat":
    current_course_id = st.session_state.selected_course_id
    st.title(f"{current_course_id} Learning Buddy")

    # Button to go back to scourse selection
    if st.sidebar.button("Change Course"):
        st.session_state.current_view = "selection"
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            # --- Cached rewriting ---
            rewritten_query = rewrite_query(prompt, st.session_state.messages, llm)

            # --- Cached search ---
            search_cache_key = hashlib.sha256((prompt + current_course_id).encode()).hexdigest()
            cached_results = get_from_cache("search_cache", search_cache_key)

            if cached_results:
                context_chunks = cached_results
            else:
                collection_name = COURSE_COLLECTIONS[current_course_id]
                query_embed = embed_model.get_query_embedding(prompt)
                results = client.search(
                    collection_name=collection_name,
                    data=[query_embed],
                    anns_field="embedding",
                    search_params={'metric_type': 'COSINE', 'params': {'nprobe': 10}},
                    output_fields=['context'],
                    limit=5,
                )
                context_chunks = [hit['entity']['context'] for hit in results[0]]
                set_cache("search_cache", search_cache_key, context_chunks)

            full_context = "\n".join(context_chunks)

            system_prompt = ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a helpful and approachable course assistant for HWU students. "
                    "Your goal is to answer questions using ONLY the provided CONTEXT. "
                    "This CONTEXT is in Markdown format. "
                    "First, identify the key FACTS from the CONTEXT that directly address the user's query. "
                    "Then, use those FACTS to construct your final answer. "
                    "If the CONTEXT does not contain enough information to answer, respond with: 'I don’t know based on the available course information.' \n"
                    "Do not generate advice, instructions, or help unrelated to the retrieved context."
                    "Do not assist with assignments, essays, reports, quizzes, or courseworks"
                    "Keep answers concise and factual.\n\n"
                    f"Course: {current_course_id}\n"
                    f"Original query: {prompt}\n"
                    # f"Rewritten query: {rewritten_query}\n\n"
                    f"CONTEXT:\n{full_context}"
                )
            )

            # Reconstruct full history for LLM
            full_chat_history = [system_prompt] + [
                ChatMessage(role=m['role'], content=m['content']) for m in st.session_state.messages
            ]
            
            # Pass full conversation history to LLM
            response = llm.chat(full_chat_history)

        with st.chat_message("assistant"):
            st.markdown(response.message.content)
            st.session_state.messages.append({"role": "assistant", "content": response.message.content})