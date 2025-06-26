#!/usr/bin/env python3
"""
RAG-Anything CLI (Simple, Neo4j+Postgres only)

- Uses .env for all storage config (see below).
- Prompts for Postgres and Neo4j credentials if not set in .env.
- Only supports Neo4j for graph DB and Postgres for all others.

"""

import os
import asyncio
from raganything.raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from dotenv import load_dotenv

load_dotenv()

def ensure_env(var, prompt):
    val = os.environ.get(var)
    while not val:
        val = input(f"{prompt}: ").strip()
        os.environ[var] = val
    return val

def main():
    print("RAG-Anything CLI (Neo4j+Postgres, remote DB credentials required)")
    print("---------------------------------------------------------------")
    print("Provide remote Neo4j and Postgres credentials.")

    ensure_env("POSTGRES_USER", "Postgres user")
    ensure_env("POSTGRES_PASSWORD", "Postgres password")
    ensure_env("POSTGRES_DATABASE", "Postgres database")
    ensure_env("POSTGRES_HOST", "Postgres host (remote, e.g. db.example.com)")
    ensure_env("POSTGRES_PORT", "Postgres port (remote, e.g. 5432)")

    ensure_env("NEO4J_URI", "Neo4j bolt URI (e.g. bolt://host:port)")
    ensure_env("NEO4J_USERNAME", "Neo4j username")
    ensure_env("NEO4J_PASSWORD", "Neo4j password")

    input_dir = input("Input directory to process (leave blank if not needed): ").strip()
    input_file = input("Input file to process (leave blank if not needed): ").strip()
    if not input_dir and not input_file:
        print("You must specify at least an input directory or input file.")
        return

    lightrag = LightRAG(
        llm_model_func=openai_complete_if_cache,
        embedding_func=openai_embed,
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE"),
        addon_params={"language": os.getenv("SUMMARY_LANGUAGE")},
    )

    rag = RAGAnything(lightrag=lightrag)

    async def run_pipeline():
        if input_file:
            await rag.process_document_complete(
                file_path=input_file,
                output_dir=os.environ.get("OUTPUT_DIR", "./output"),
                parse_method=os.environ.get("PARSE_METHOD", "auto"),
                display_stats=False,
            )
        if input_dir:
            await rag.process_folder_complete(
                folder_path=input_dir,
                output_dir=os.environ.get("OUTPUT_DIR", "./output"),
                parse_method=os.environ.get("PARSE_METHOD", "auto"),
                display_stats=False,
                max_workers=int(os.environ.get("MAX_WORKERS", 2)),
            )

    try:
        asyncio.run(run_pipeline())
        print("\n✅ Processing complete! Data has been inserted into vectorDB and graphDB.")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 