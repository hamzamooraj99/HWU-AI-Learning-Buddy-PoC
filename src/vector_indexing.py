import os
import json
from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "HWU_learning_buddy_prototype"

def create_collection(client: MilvusClient, collection_name=COLLECTION_NAME):
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_fieds=True
    )

    # Add a primary key field
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

    # Add other fields
    schema.add_field("course_id", DataType.VARCHAR, max_length=5)
    schema.add_field("context", DataType.VARCHAR, max_length=2000)  
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)
    # NOTE: "bge-small-en-v1.5" → 384 dims; if swap to bge-large, change to 1024

    # Define index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="IVF_FLAT",
        metric_type="COSINE"
    )

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print(f"[INFO] Created collection '{collection_name}' with schema + index")

def insert_embeddings(client, input_file, collection_name=COLLECTION_NAME):
    with open(input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    data = []
    for rec in records:
        text = rec.get('text') or rec.get('content')
        embedding = rec.get('embedding')
        if embedding is None:
            continue
        data.append({
            'context': text,
            'embedding': embedding,
            'course_id': rec.get('metadata', {}).get('course_id')
        })
    
    client.insert(collection_name=collection_name, data=data)
    print(f"[INFO] Inserted {len(data)} records into '{collection_name}'")

if __name__ == '__main__':
    this_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))

    client = MilvusClient(
        uri="http://localhost:19530"
    )

    embeddings_file = os.path.join(root_dir, "data", "F21CA_embeddings.json")

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"[INFO] Dropped old collection: {COLLECTION_NAME}")

    create_collection(client)
    insert_embeddings(client, embeddings_file)