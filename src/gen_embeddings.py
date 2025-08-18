import os
import json
import numpy as np
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def generate_embeddings(input_file, output_file, batch_size=32):
    # Load ingested data
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[INFO] Loaded {len(records)} records from {input_file}")

    # Choose embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",  # swap to large if resources allow
        trust_remote_code=True,
        cache_folder="./hf_cache"
    )

    # Extract text
    texts = [rec["text"] if "text" in rec else rec["content"] for rec in records]

    # Generate embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_embeds = embed_model.get_text_embedding_batch(batch)
        embeddings.extend(batch_embeds)

    # Attach embeddings to records
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = emb  # keep as float32 for now

    # Save with embeddings
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Saved {len(records)} embeddings → {output_file}")

if __name__ == "__main__":
    # Work out repo root = parent of src/
    this_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))

    input_path = os.path.join(root_dir, "data", "F21CA_site_data.json")
    output_path = os.path.join(root_dir, "data", "F21CA_embeddings.json")

    generate_embeddings(input_path, output_path)
