"""
Environment variable utilities for the RAG-Anything pipeline.
Ensures all required environment variables are set, prompting the user if necessary.
"""

import os


def ensure_env_vars():
    """
    Ensure all required environment variables are set, prompting the user for any missing values.
    """
    env_var_prompts = [
        ("OPENAI_API_KEY", "OpenAI API key"),
        ("POSTGRES_USER", "Postgres user"),
        ("POSTGRES_PASSWORD", "Postgres password"),
        ("POSTGRES_DATABASE", "Postgres database"),
        ("POSTGRES_HOST", "Postgres host (remote, e.g. db.example.com)"),
        ("POSTGRES_PORT", "Postgres port (remote, e.g. 5432)"),
        ("NEO4J_URI", "Neo4j bolt URI (e.g. bolt://host:port, bolt://neo4j:7687)"),
        ("NEO4J_USERNAME", "Neo4j username"),
        ("NEO4J_PASSWORD", "Neo4j password"),
        ("INPUT_DIR", "Input directory to process (leave blank if not needed)"),
        ("INPUT_FILE", "Input directory to process (leave blank if not needed)")
    ]
    for var_name, prompt in env_var_prompts:
        value = os.environ.get(var_name)
        while not value:
            value = input(f"{prompt}: ").strip()
            os.environ[var_name] = value
