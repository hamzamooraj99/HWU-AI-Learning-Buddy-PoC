from beam import Image, Pod

streamlit_server = Pod(
    image=Image().add_python_packages([
        "streamlit",
        "pymilvus",
        "llama-index",
        "llama-index-embeddings-huggingface",
        "llama-index-llms-groq"
    ]),
    ports=[8501],  # Default port for Streamlit
    gpu="T4",
    memory="2Gi",
    entrypoint=["streamlit", "run", "app.py"],
)

res = streamlit_server.create()

# New way to get deployment URL
print("Deployed at:", res.url)
