"""
Pipeline utilities for initializing and running the RAG-Anything processing pipeline with cross-modal relationship support.
Handles input collection, RAGAnything initialization, and enhanced pipeline execution with automatic cross-modal relationship mapping.
"""

import os
import traceback
from raganything.raganything import RAGAnything
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed


def init_raganything(llm_model_func, vision_model_func):
    """
    Initialize the RAGAnything object with LightRAG and model functions, including cross-modal relationship support.
    Args:
        llm_model_func (callable): Function for LLM model invocation.
        vision_model_func (callable): Function for vision model invocation.
    Returns:
        RAGAnything: Initialized RAGAnything instance with cross-modal relationship capabilities.
    """
    print("[Pipeline] 🔧 Initializing RAGAnything with cross-modal relationship support...")
    
    lightrag_instance = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM")),
            max_token_size=int(os.getenv("MAX_EMBEDDING_TOKEN_SIZE")),
            func=lambda texts: openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        ),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE"),
        addon_params={"language": os.getenv("SUMMARY_LANGUAGE")},
    )
    
    rag = RAGAnything(
        lightrag=lightrag_instance,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM")),
            max_token_size=int(os.getenv("MAX_EMBEDDING_TOKEN_SIZE")),
            func=lambda texts: openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        ),
    )
    
    print("[Pipeline] ✅ RAGAnything initialized with cross-modal relationship support")
    print("[Pipeline] 🔗 Cross-modal relationship manager ready")
    return rag


async def initialize_pipeline_status():
    """
    Initialize the pipeline status using LightRAG's shared storage.
    Returns True if successful, False otherwise.
    """
    print("[Pipeline] 🔧 Initializing pipeline status...")
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status as init_status

        await init_status()
        print("[Pipeline] ✅ Pipeline status initialized")
        return True
    except Exception as error:
        print(f"[Pipeline] ⚠️  Warning: Could not initialize pipeline status: {error}")
        return False


async def process_input_file(rag, input_file):
    """
    Process a single input file using the enhanced RAG pipeline with cross-modal relationship support.
    """
    print(f"[Pipeline] 📄 Processing input file: {input_file}")
    print(f"[Pipeline] 🔄 Enhanced pipeline with cross-modal relationship inference")
    
    try:
        result = await rag.process_document_complete(
            file_path=input_file, display_stats=True
        )
        print(f"[Pipeline] ✅ Finished processing file: {input_file}")
        print(f"[Pipeline] 📊 File processing result: {result}")
        print(f"[Pipeline] 🔗 Cross-modal relationships automatically created")
        
    except Exception as e:
        print(f"[Pipeline] ❌ Error processing file {input_file}: {e}")
        print(traceback.format_exc())


async def process_input_dir(rag, input_dir):
    """
    Process all files in the input directory using the enhanced RAG pipeline with cross-modal relationship support.
    """
    print(f"[Pipeline] 📁 Processing input directory: {input_dir}")
    print(f"[Pipeline] 🔄 Enhanced pipeline with cross-modal relationship inference")
    
    try:
        result = await rag.process_folder_complete(
            folder_path=input_dir, display_stats=True
        )
        print(f"[Pipeline] ✅ Finished processing directory: {input_dir}")
        print(f"[Pipeline] 📊 Directory processing result: {result}")
        print(f"[Pipeline] 🔗 Cross-modal relationships automatically created for all files")
        
    except Exception as e:
        print(f"[Pipeline] ❌ Error processing directory {input_dir}: {e}")
        print(traceback.format_exc())


async def run_pipeline(rag, input_file, input_dir):
    """
    Orchestrate the enhanced RAG pipeline with cross-modal relationship support for the provided input file and/or directory.
    """
    print("[Pipeline] 🚀 Starting enhanced pipeline run with cross-modal relationship support...")
    print(f"[Pipeline] 📄 Input file: {input_file if input_file else 'None'}")
    print(f"[Pipeline] 📁 Input directory: {input_dir if input_dir else 'None'}")
    print("[Pipeline] 🔗 Cross-modal relationship inference enabled")

    await initialize_pipeline_status()

    if input_file:
        await process_input_file(rag, input_file)
    else:
        print("[Pipeline] ℹ️  No input file provided.")

    if input_dir:
        await process_input_dir(rag, input_dir)
    else:
        print("[Pipeline] ℹ️  No input directory provided.")

    print("[Pipeline] ✅ Enhanced pipeline run complete with cross-modal relationships.")
    print("[Pipeline] 💡 You can now query cross-modal relationships in your knowledge graph!")
