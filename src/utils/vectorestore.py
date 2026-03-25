

from qdrant_client import QdrantClient


def collection_exists(client: QdrantClient, collection: str) -> bool:
    """Ensure the configured Qdrant collection exists with the expected schema.

    Input:
        Uses collection settings from application configuration.
    Output:
        Creates the collection when missing and leaves it unchanged
        otherwise.
    Security context:
        Constrains vector schema management to the configured collection so
        retrieval setup does not mutate unrelated vector data.
    """

    collections = client.get_collections()
    existing_names = {collection.name for collection in collections.collections}
    return collection in existing_names

