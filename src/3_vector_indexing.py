import os
import json
from pymilvus import MilvusClient, DataType

def create_collection(client: MilvusClient, collection_name: str):
    """
    Creates a new collection with a defined schema and an index for vector search
    """
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_fields=True
    )

    # Add a primary key field - unique for each record
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

    # Add other fields
    schema.add_field("course_id", DataType.VARCHAR, max_length=5)
    schema.add_field("context", DataType.VARCHAR, max_length=8192)  # Increased max_length to 8192
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

def insert_embeddings(client, input_file, collection_name: str):
    """
    Inserts data from a JSON file into the specified Milvus collection
    """
    if not os.path.exists(input_file):
        print(f"[WARN] Input file not found: {input_file}. Skipping...")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    data = []
    for rec in records:
        text = rec.get('text') or rec.get('content')
        embedding = rec.get('embedding')
        course_id = rec.get('metadata', {}).get('course_id')
        if embedding is None or course_id is None:
            continue

        data.append({
            'context': text,
            'embedding': embedding,
            'course_id': course_id
        })
    
    if data:
        client.insert(collection_name=collection_name, data=data)
        print(f"[INFO] Inserted {len(data)} records into '{collection_name}'")
    else:
        print(f"[WARN] No records with embeddings found in {input_file}. No data inserted.")


if __name__ == '__main__':
    this_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))

    client = MilvusClient(
        uri="http://milvus_db:19530"
    )

    course_ids = ['F21CA', 'F21NL']

    for course_id in course_ids:
        collection_name = f"HWU_MACS_{course_id}"
        embeddings_file = os.path.join(root_dir, "data", f"{course_id}_embeddings.json")

        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"[INFO] Dropped old collection: {collection_name}")

        create_collection(client, collection_name)
        insert_embeddings(client, embeddings_file, collection_name)