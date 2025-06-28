"""
Entry point for the RAG-Anything CLI application.
Initializes environment, loads models, and runs the processing pipeline.
"""

import os

from dotenv import load_dotenv
from utils.environment import ensure_env_vars
from utils.model_invocation import (
    llm_model_func,
    vision_model_func,
)
from pipeline import init_raganything, run_pipeline
import asyncio


def main():
    """
    Main function to initialize environment, prompt for credentials, load models, and run the processing pipeline.
    """
    load_dotenv()
    print("RAG-Anything CLI (Neo4j+Postgres, remote DB credentials required)")
    print("---------------------------------------------------------------")
    print("Provide remote Neo4j and Postgres credentials.")
    ensure_env_vars()
    rag = init_raganything(llm_model_func, vision_model_func)
    input_dir = os.getenv("INPUT_DIR")
    input_file = os.getenv("INPUT_FILE")
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_pipeline(rag, input_file, input_dir))
        print(
            "\n✅ Processing complete! Data has been inserted into vectorDB and graphDB."
        )
    except Exception as error:
        print(f"\n❌ Error during processing: {error}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
