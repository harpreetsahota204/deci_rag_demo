
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, NoReturn
import intel_extension_for_pytorch as ipex
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.huggingface import HuggingFaceLLM

from ingest import setup

setup()

client = qdrant_client.QdrantClient(
    path="/home/demotime/DeciLM_RAG_Demo/vector_store",
    prefer_grpc=True
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="SuperMicro Solutions Briefs",
    )

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)