


import asyncio

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import AppConfig
from core.factory import get_collection_name, get_embedding_dimension, get_embeddings
from ingestion.harvester import NvdHarvester
from utils.vectorestore import collection_exists


def _distance_enum(distance: str) -> Distance:
    """Translate configured distance text into a Qdrant distance enum.

    Input:
        Uses the configured Qdrant distance metric string.
    Output:
        Returns the matching Qdrant distance enum.
    Security context:
        Restricts collection schema creation to approved distance metrics.
    """
    mapping = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
        "manhattan": Distance.MANHATTAN,
    }
    if distance not in mapping:
        raise ValueError("Unsupported Qdrant distance metric: {value}".format(value=distance))
    return mapping[distance]


async def build_nvd_harvester(config: AppConfig) -> NvdHarvester:
    qclient = QdrantClient(url=config.qdrant_url)
    collection_name = get_collection_name(prefix=config.qdrant_collection)

    # Ensure collection exists
    if not collection_exists(qclient, collection_name):
        await asyncio.to_thread(
            qclient.create_collection,
            collection_name=collection_name, 
            vectors_config={
                config.qdrant_vector_name: VectorParams(
                    size=config.qdrant_vector_size or get_embedding_dimension(),
                    distance=_distance_enum(config.qdrant_distance),
                )
            }
        )

    # Create Qdrant vector store
    vectorstore = QdrantVectorStore(
        qclient, 
        collection_name,
        embedding=get_embeddings(),
        vector_name=config.qdrant_vector_name,
    )

    return NvdHarvester(vectorstore=vectorstore)

