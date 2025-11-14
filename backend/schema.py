# schema.py
from weaviate_client import client
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.config import ConsistencyLevel

COLLECTION_NAME = "ApologyCase"

def ensure_schema():
    """
    Create the ApologyCase collection if it doesn't exist.
    """
    try:
        client.collections.get(COLLECTION_NAME)
        # already exists
        return
    except weaviate.exceptions.WeaviateBaseError:
        pass

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_huggingface(),  # module must be enabled in WCS
        properties=[
            Property(name="case_name", data_type=DataType.TEXT),
            Property(name="raw_text", data_type=DataType.TEXT),
        ],
        consistency_level=ConsistencyLevel.QUORUM,
    )


def reset_collection():
    """
    Delete all objects in the ApologyCase collection (called when uploading a new PDF).
    """
    coll = client.collections.get(COLLECTION_NAME)
    coll.data.delete_many(where={})
