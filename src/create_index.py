import os
import json
import numpy as np

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

def create_milvus_index(course_id, milvus_host="localhost", milvus_port="19530"):
    """
    Creates and populates a Milvus index with data for a specific course.
    
    Args:
        course_id (str): The ID of the course to index.
        milvus_host (str): The host address of the Milvus server.
        milvus_port (str): The port of the Milvus server.
    """
    data_file = os.path.join("data", f"{course_id}_site_data.json")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    # 1. Load the data from the JSON file
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data_records = json.load(f)
    
    # 2. Initialize the embedding model
    print("Initializing embedding model...")
    embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # 3. Create a Milvus vector store with a binary index
    print("Connecting to Milvus and creating collection...")
    vector_store = MilvusVectorStore(
        uri=f"tcp://{milvus_host}:{milvus_port}",
        collection_name=f"course_data_{course_id}",
        overwrite=True,  # Overwrite if the collection already exists
        dim=384,         # This is the dimension of the BAAI/bge-small-en-v1.5 model
        metric_type="HAMMING" # Use Hamming distance for binary vectors
    )

    # 4. Process and insert data into Milvus
    print("Generating embeddings and inserting data...")
    nodes = []
    for record in data_records:
        text = record["text"]
        metadata = record["metadata"]
        
        # Generate the embedding using llama_index
        embedding = embed_model.get_text_embedding(text)
        
        # Convert the float32 vector to a binary vector (Quantization)
        # We use a simple numpy approach for this demonstration
        binary_embedding = (np.array(embedding) > np.mean(embedding)).astype(np.bool_).tobytes()

        # Create a TextNode object for llama_index
        node = TextNode(
            text=text,
            embedding=binary_embedding,
            metadata=metadata
        )
        nodes.append(node)

    # The MilvusVectorStore handles the final insertion
    vector_store.add(nodes)

    print(f"Successfully indexed {len(nodes)} records into Milvus.")

if __name__ == "__main__":
    # Example usage: you must have a Milvus server running for this to work
    create_milvus_index(course_id="F21CA")