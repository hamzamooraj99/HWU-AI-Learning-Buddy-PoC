import streamlit as st
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# --- Setup connections ---
COLLECTION_NAME = "HWU_learning_buddy_prototype"
milvus_uri = "http://localhost:19530"   # or your Milvus service
client = MilvusClient(uri=milvus_uri)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Groq(
    model="moonshotai/kimi-k2-instruct",
    api_key="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",
    temperature=0.5,
    max_tokens=800,
)

# --- Streamlit UI ---
st.title("HWU AI Learning Buddy 🤖")
# Initialize chat history in Streamlit's session state for multi-turn conversations
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
query = st.text_input("Enter your question:")

if st.button("Ask") and query.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):

        # Step 4: Retrieval
        query_embed = embed_model.get_query_embedding(query)
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embed],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            output_fields=["context"],
            limit=5,
        )

        context_chunks = [hit["entity"]["context"] for hit in results[0]]
        full_context = "\n".join(context_chunks)

        # Step 5: Generation
        prompt_template = (
            "Context information is below.\n"
            "---------------------\n"
            "CONTEXT: {context}\n"
            "---------------------\n"
            "Given the context information above, "
            "answer the user's query concisely.\n"
            "QUERY: {query}\n"
            "ANSWER: "
        )
        prompt = prompt_template.format(context=full_context, query=query)

        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)

        response_gen = llm.stream_complete(user_msg.content)

        # Display as streaming output
        answer_placeholder = st.empty()
        full_answer = ""
        for chunk in response_gen:
            full_answer += chunk.delta
            answer_placeholder.markdown(full_answer)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
