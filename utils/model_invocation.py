"""
Model invocation utilities for the RAG-Anything pipeline.
Provides functions to obtain LLM, vision, and embedding model callables.
"""

import os

from lightrag.llm.openai import openai_complete_if_cache


def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """
    Invoke the LLM model with the given prompt and optional parameters.
    Args:
        prompt (str): The user prompt.
        system_prompt (str, optional): The system prompt.
        history_messages (list, optional): Conversation history.
        **kwargs: Additional keyword arguments for the model.
    Returns:
        str: Model response.
    """
    if history_messages is None:
        history_messages = []
    return openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )


def vision_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    image_data=None,
    **kwargs,
):
    """
    Invoke the vision model with the given prompt, image data, and optional parameters.
    Args:
        prompt (str): The user prompt.
        system_prompt (str, optional): The system prompt.
        history_messages (list, optional): Conversation history.
        image_data (str, optional): Base64-encoded image data.
        **kwargs: Additional keyword arguments for the model.
    Returns:
        str: Model response.
    """
    if history_messages is None:
        history_messages = []
    if image_data:
        return openai_complete_if_cache(
            "gpt-4o",
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                (
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None
                ),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ],
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )
    return openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )
