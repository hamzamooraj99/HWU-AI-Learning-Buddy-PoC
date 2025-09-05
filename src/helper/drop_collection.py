from pymilvus import MilvusClient

client = MilvusClient(
    host="milvus_db",
    port="19530"
)


client.drop_collection("HWU_MACS_F21CA")