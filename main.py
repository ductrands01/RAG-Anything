"""
Entry point for the RAG-Anything CLI application.
Initializes environment, loads models, and runs the processing pipeline with cross-modal relationship support.
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
    Main function to initialize environment, prompt for credentials, load models, and run the processing pipeline with cross-modal relationship support.
    """
    load_dotenv()
    print("🚀 RAG-Anything CLI with Cross-Modal Relationship Support")
    print("=" * 60)
    print("🔗 Enhanced with automatic cross-modal relationship mapping")
    print("📊 Supports relationships between text, images, tables, and equations")
    print("=" * 60)
    print("Provide remote Neo4j and Postgres credentials.")
    ensure_env_vars()
    
    print("\n🔧 Initializing RAG-Anything with cross-modal relationship support...")
    rag = init_raganything(llm_model_func, vision_model_func)
    
    input_dir = os.getenv("INPUT_DIR")
    input_file = os.getenv("INPUT_FILE")
    
    print(f"\n📁 Input Configuration:")
    print(f"   - Input File: {input_file if input_file else 'None'}")
    print(f"   - Input Directory: {input_dir if input_dir else 'None'}")
    
    try:
        loop = asyncio.get_event_loop()
        print("\n🔄 Starting enhanced processing pipeline...")
        print("   - Step 1: Document parsing with MinerU")
        print("   - Step 2: Text content processing")
        print("   - Step 3: Multimodal content processing")
        print("   - Step 4: Cross-modal relationship inference ✨")
        print("   - Step 5: Knowledge graph integration")
        
        loop.run_until_complete(run_pipeline(rag, input_file, input_dir))
        
        print("\n✅ Processing complete!")
        print("📊 Data has been inserted into vectorDB and graphDB.")
        print("🔗 Cross-modal relationships have been automatically created.")
        print("\n💡 You can now query cross-modal relationships like:")
        print("   - 'What images illustrate the Catalog Management section?'")
        print("   - 'Which tables explain the Product Information?'")
        print("   - 'Show me all entities that contain images'")
        
    except Exception as error:
        print(f"\n❌ Error during processing: {error}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
