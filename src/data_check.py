from pymilvus import connections, utility

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Collections to drop
collections = ["HWU_learning_buddy_prototype", "course_data_F21CA"]

for c in collections:
    if utility.has_collection(c):
        utility.drop_collection(c)
        print(f"Dropped collection: {c}")
    else:
        print(f"Collection not found: {c}")

print(utility.list_collections())
