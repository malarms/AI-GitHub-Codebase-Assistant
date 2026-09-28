from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorDB:

    def __init__(self, collection_name="code_chunks"):

        self.client = QdrantClient(":memory:")

        self.collection_name = collection_name

    def create_collection(self, vector_size: int):

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