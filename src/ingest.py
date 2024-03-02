import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, NoReturn

from llama_index.core import  Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.extractors import QuestionsAnsweredExtractor, SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

import cleaning_utils

import setup_utils
    
def create_documents_from_clean_text(cleaned_texts: List[Tuple[str, Dict]]) -> List[Document]:
    """
    Constructs and returns a list of Document objects from cleaned text and metadata.

    This function takes a list of tuples, where each tuple contains cleaned text and its
    associated metadata dictionary. It creates a Document object for each tuple, with the
    text set to the cleaned text and metadata set to the associated dictionary. These Document
    objects are assumed to be part of the `llama_index` module, designed to represent processed
    text documents in a structured form.

    Parameters:
    ----------
    cleaned_texts : List[Tuple[str, Dict]]
        A list of tuples, where each tuple contains a string of cleaned text and a dictionary
        of metadata related to that text. The metadata dictionary can include any relevant
        information such as publication date, referenced websites, file names, etc.

    Returns:
    -------
    List[Document]
        A list of Document objects, where each Document corresponds to an entry in the input list,
        encapsulating the cleaned text and its associated metadata. These Document objects are suitable
        for indexing, analysis, or any further processing within the `llama_index` framework.

    """
    documents = [Document(text=t, 
                          metadata=m, 
                          metadata_seperator="\n\n", 
                          excluded_llm_metadata_keys=["file_name",
                                                      "publication_date", 
                                                      "referenced_websites", 
                                                      "section_summary", 
                                                      "excerpt_keywords",
                                                      "questions_this_excerpt_can_answer"
                                                     ]
                         ) for (t, m) in cleaned_texts]
    return documents

def create_metadata_extractors():
    qa_prompt = """ Here is the contextual information from a solution brief by SuperMicro:
    {context_str}

    Given the contextual information, generate {num_questions} to the point questions this context can provide \
    specific answers about the products, software, hardware, and solutions discussed in this document\
    which are unlikely to be found elsewhere.

    Higher-level summaries of the surrounding context may be provided as well.  Try using these summaries to generate better questions that this context can answer.\
    Ensure that your questions are detailed, yet to the point, and are about the specific products, software, hardware, and solutions discussed in this document"""

    summary_prompt = """ Here is the content of the section, which is from a solution brief by SuperMicro:

    {context_str}

    Provide a Summary of the section. Also, identify the specific companies, technology products, software, hardware (GPUs, CPUs, memory, accelerators, etc), and solutions discussed in this section.

    Summary: 

    """
    text_splitter = TokenTextSplitter(
        separator=" ", 
        chunk_size=612, 
        chunk_overlap=8
    )

    qa_extractor = QuestionsAnsweredExtractor(
        questions=5, 
        prompt_template=qa_prompt,
        num_workers=os.cpu_count()
    )

    summary = SummaryExtractor(
        summaries = ["self"], 
        prompt_template=summary_prompt,
        num_workers=os.cpu_count()
    )

    # key_words = KeywordExtractor(
    #     keywords=5,
    #     num_workers=os.cpu_count()
    # )
    
    extractors = [text_splitter, summary, qa_extractor]
    
    return extractors
    
def build_pipeline(transforms):
    return IngestionPipeline(transformations=transforms)

def build_nodes(documents, pipeline, transforms):
    pipeline = pipeline
    nodes = pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True
        )
    return nodes

def create_vector_store(nodes, 
                        persist_path: str = "/home/demotime/DeciLM_RAG_Demo/vector_store",
                        collection_name: str = "SuperMicro Solutions Briefs") -> NoReturn:
    """
    Initializes and persists a vector store with a specified collection of nodes.

    This function creates a Qdrant vector store and a collection within it using the provided nodes.
    It configures the vector store client, sets up the collection with the given name, and persists
    the entire vector store at the specified path. The vector store is intended to facilitate
    efficient storage and retrieval of vector embeddings for the nodes.

    Parameters:
    ----------
    nodes : List[Node]
        A list of nodes (documents, embeddings, etc.) to be stored in the vector store.
    persist_path : str, optional
        The filesystem path where the vector store will be persisted. Defaults to a predefined path.
    collection_name : str, optional
        The name of the collection within the vector store. Defaults to 'SuperMicro Solutions Briefs'.

    Returns:
    -------
    NoReturn
        This function does not return any value. It creates and persists the vector store as a side effect.
    """
    client = qdrant_client.QdrantClient(path=persist_path)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=collection_name,
        path=persist_path
        )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_path)

    
def main() -> NoReturn:
    """Main function to orchestrate the document processing pipeline."""
    
    # Path to the directory containing documents to be processed
    documents_dir = Path('/home/demotime/DeciLM_RAG_Demo/SuperMicro_Solution_Brief')   
    
    # Clean and prepare texts
    cleaned_texts = cleaning_utils.clean_and_prepare_texts(documents_dir)
    logging.info(f"Cleaned and prepared {len(cleaned_texts)} documents.") 
    
    # Setup the environment and models
    setup_utils.setup_llm()
    logging.info(f"LLM setup complete, using {Settings.llm.model_name}")
    
    setup_utils.embed_model()
    logging.info(f"Embedding model setup complete, using {Settings.embed_model.model_name}")

    # Create metadata extractors and build the ingestion pipeline
    metadata_extractors = create_metadata_extractors()
    pipeline = build_pipeline(metadata_extractors)
    logging.info("Ingestion pipeline built.")
    
    # Create documents from cleaned texts
    documents = create_documents_from_clean_text(cleaned_texts)
    
    # Process documents through the pipeline and create vector store
    nodes = build_nodes(documents, pipeline, metadata_extractors)
    create_vector_store(nodes)
    logging.info("Vector store created successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()