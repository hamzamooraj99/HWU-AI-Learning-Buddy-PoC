import os
import json
import numpy as np
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# -----------------------------
# CONFIG
# -----------------------------
COLLECTION_NAME = "HWU_learning_buddy_prototype"
TOP_K = 5  # number of chunks to retrieve
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # must match Step 2 embeddings
LLM_MODEL = "moonshotai/kimi-k2-instruct"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # set this in your env

# -----------------------------
# FUNCTIONS
# -----------------------------
def binary_quantize(embeddings):
    """Convert float32 embeddings to binary vectors (0/1)"""
    binary = np.where(np.array(embeddings) > 0, 1, 0).astype(np.uint8)
    packed = np.packbits(binary, axis=1)
    return [vec.tobytes() for vec in packed]

def retrieve_context(client, query_embedding, collection_name, top_k=TOP_K):
    """Search Milvus for top_k relevant chunks"""
    query_vector = query_embedding

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        search_params={"metric_type": "COSINE"},
        output_fields=["context"],
        limit=top_k
    )

    full_context = []
    for res in results:
        top_hit = res[0]
        context = top_hit.entity.get("context")
        full_context.append(context)
    return full_context

def generate_answer(context_chunks, query):
    """Send context + query to Kimi-K2 LLM"""
    llm = Groq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=1000
    )

    prompt_template = (
        "Context information is below.\n"
        "---------------------\n"
        "CONTEXT: {context}\n"
        "---------------------\n"
        "Given the context information above, answer the user's query "
        "in a crisp and concise manner. If unknown, say 'I don't know!'.\n"
        "QUERY: {query}\n"
        "ANSWER: "
    )

    prompt = prompt_template.format(context="\n".join(context_chunks), query=query)
    user_msg = ChatMessage(role=MessageRole.USER, content=prompt)

    # Streamed response from LLM
    response = llm.stream_complete(user_msg.content)
    return response

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Connect to Milvus
    client = MilvusClient(uri="http://localhost:19530")

    # Load embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        trust_remote_code=True,
        cache_folder="./hf_cache"
    )

    # Example user query
    user_query = input("Enter your query: ")

    # Step 4: Embed query
    query_embed = embed_model.get_query_embedding(user_query)

    # Step 4: Retrieve relevant context
    context = retrieve_context(client, query_embed, COLLECTION_NAME)

    # Step 5: Generate answer
    answer = generate_answer(context, user_query)
    print("\n=== ANSWER ===")
    print(answer)
