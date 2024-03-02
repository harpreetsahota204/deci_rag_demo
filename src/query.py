import logging
import warnings
from pathlib import Path
from typing import List, Union

# Suppress all warnings
warnings.filterwarnings("ignore")

# import intel_extension_for_pytorch as ipex
import qdrant_client
from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, get_response_synthesizer)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

import setup_utils


def get_nodes_from_vector_store(index: VectorStoreIndex) -> List:
    """
    Retrieves nodes from the vector store based on a predefined query.

    Parameters:
    - index: The vector store index object.

    Returns:
    - List of nodes retrieved from the vector store.
    """
    source_nodes = index.as_retriever(similarity_top_k=1000000).retrieve("SuperMicro")
    nodes = [x.node for x in source_nodes]
    return nodes 


def create_bm25_retriever(nodes: List, similarity_top_k: int = 15) -> BM25Retriever:
    """
    Creates a BM25Retriever instance.

    Parameters:
    - nodes: List of nodes to be used by the retriever.
    - similarity_top_k: The number of top similar items to retrieve.

    Returns:
    - An instance of BM25Retriever.
    """
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)
    return retriever


def create_vector_retriever(index: VectorStoreIndex, similarity_top_k: int = 15) -> VectorIndexRetriever:
    """
    Creates a VectorIndexRetriever instance.

    Parameters:
    - index: The vector store index object.
    - similarity_top_k: The number of top similar items to retrieve.

    Returns:
    - An instance of VectorIndexRetriever.
    """
    vector_retriever = VectorIndexRetriever(index, similarity_top_k, alpha=0.65)
    return vector_retriever


class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines BM25 and vector-based retrieval methods.
    """
    def __init__(self, vector_retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever):
        """
        Initializes the HybridRetriever.

        Parameters:
        - vector_retriever: An instance of VectorIndexRetriever.
        - bm25_retriever: An instance of BM25Retriever.
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        """
        Retrieves and combines results from both BM25 and vector retrievers.

        Parameters:
        - query: The query string.

        Returns:
        - A combined list of nodes from both retrievers.
        """
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

def create_reranker(
    top_n :int = 5,
    model:str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
    device:str = "xpu"):
    """
    Creates a reranker based on the SentenceTransformer model.

    Parameters:
    - top_n: The number of top items to rerank.
    - model: The model name or path.
    - device: The device to run the model on.

    Returns:
    - An instance of SentenceTransformerRerank.
    """
    reranker = SentenceTransformerRerank(top_n=top_n, model=model, device=device)
    return reranker

def create_query_engine(retriever, post_processors:List):
    """
    Creates a query engine with specified retriever and post-processors.

    Parameters:
    - retriever: The retriever instance (HybridRetriever or other).
    - post_processors: A list of post-processor instances.

    Returns:
    - An instance of RetrieverQueryEngine.
    """   
    synth = get_response_synthesizer(streaming=True)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=post_processors,
        response_synthesizer=synth
    )
    return query_engine

import argparse

def initialize_system():
    """
    Initializes the chatbot system by setting up models and indices.
    Returns the initialized query engine.
    """
    setup_utils.setup_llm()
    setup_utils.setup_embed_model()
    index = setup_utils.setup_index()
    
    nodes = get_nodes_from_vector_store(index)
    bm25_retriever = create_bm25_retriever(nodes, 7)
    vector_retriever = create_vector_retriever(index)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    reranker = create_reranker()
    query_engine = create_query_engine(hybrid_retriever, [reranker])
    return query_engine

def handle_query(query_engine):
    """
    Enters an interactive mode where the user can type queries, and the system responds.
    Loops until the user types 'exit'.
    """
    print("Enter your query or type 'exit' to quit:")
    while True:
        user_input = input("👨>")
        if user_input.lower() == 'exit':
            break
        query_engine.query(user_input).print_response_stream()

def main():
    parser = argparse.ArgumentParser(description="Chatbot System")
    parser.add_argument("--mode", choices=['init', 'query'], help="Choose 'init' to initialize and 'query' to query the system.")
    args = parser.parse_args()

    if args.mode == 'init':
        query_engine = initialize_system()
        print("System initialized. You can now start querying.")
    elif args.mode == 'query':
        query_engine = initialize_system()  # Assuming you have a way to load or persist the engine state.
        handle_query(query_engine)

if __name__ == "__main__":
    main()
