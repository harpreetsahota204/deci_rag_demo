import os
from pathlib import Path
from typing import List, Tuple, Dict, NoReturn
# import intel_extension_for_pytorch as ipex

from llama_index.core import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def setup_llm(
    model_name: str = "Deci/DeciLM-7B-instruct",
    device_map: str = "xpu",
    context_window: int = 4096,
    max_length: int =1024,
    gen_kwargs: Dict = {"temperature": 0.1, "do_sample": True},
    is_chat_model: bool = True,
    tokenizer_kwargs= {"pad_token_id": 2, "eos_token_id":2},
    model_kwargs: Dict ={"torch_dtype": "auto", "trust_remote_code":True, "cache_dir":"/home/demotime/DeciLM_RAG_Demo/llms"},
    system_prompt: str = "You are an AI assistant that follows instructions extremely well. Help as much as you can."
    ) -> NoReturn:
    """
    Initializes and configures a HuggingFace language learning model (LLM) with specified parameters.

    This function sets up an LLM for use in generating responses. It configures the model
    with a given context window, generation parameters, and other settings, and stores the configured model
    in the global Settings object for later use.

    Parameters:
    - model_name (str): Name of the model to be used.
    - device_map (str): Device type for model execution (e.g., 'cpu', 'cuda', 'xpu').
    - context_window (int): Size of the context window for the model.
    - gen_kwargs (Dict): Generation parameters for the model.
    - is_chat_model (bool): Flag indicating if the model is a chat model.
    - tokenizer_kwargs (Dict): Tokenizer parameters.
    - model_kwargs (Dict): Model configuration parameters.
    - system_prompt (str): Default system prompt for the model.

    Returns:
    - NoReturn: This function does not return a value but updates the global Settings object.
    """
    llm = HuggingFaceLLM(
        context_window=context_window,
        generate_kwargs=gen_kwargs,
        is_chat_model=is_chat_model,
        system_prompt= system_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map=device_map,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs
    )
    
    Settings.llm = llm
    
def setup_embed_model(
    model_name: str = "WhereIsAI/UAE-Large-V1",
    tokenizer_name: str = "Deci/DeciLM-7B-instruct",
    device: str = "xpu",
    trust_remote_code: bool = True,
    cache_folder: str = "/home/demotime/DeciLM_RAG_Demo/embed_models"
    ) -> NoReturn:
    """
    Initializes and configures a HuggingFace embedding model with specified parameters.

    This function sets up an embedding model designed to generate or process embeddings for text data.
    It configures the model with specific parameters related to device execution, security, and caching,
    and stores the configured model in the global Settings object for later use.

    Parameters:
    - model_name (str): Name of the embedding model to be used.
    - tokenizer_name (str): Name of the tokenizer associated with the model.
    - device (str): Device type for model execution (e.g., 'cpu', 'cuda', 'xpu').
    - trust_remote_code (bool): Whether to trust remote code when loading the model.
    - cache_folder (str): Directory for caching model data.

    Returns:
    - NoReturn: This function does not return a value but updates the global Settings object.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
        trust_remote_code=trust_remote_code,
        cache_folder = cache_folder
        )
    
    Settings.embed_model = embed_model