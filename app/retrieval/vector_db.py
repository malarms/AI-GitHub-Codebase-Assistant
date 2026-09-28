from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorDB:
    def __init__(
        self,
        collection_name="code_chunks",
        storage_path="data/index/qdrant",
    ):
        self.client = QdrantClient(path=storage_path)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def insert(self, points: list[PointStruct]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query_vector, limit=5):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        ).points