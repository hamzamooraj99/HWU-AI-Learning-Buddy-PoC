from pymilvus import MilvusClient

client = MilvusClient(
        uri="http://localhost:19530"
    )


client.drop_collection("HWU_MACS_F21CA")