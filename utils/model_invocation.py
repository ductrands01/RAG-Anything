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


def get_vision_model_func(prompt, system_prompt, image_data, **kwargs):

    return lambda prompt, system_prompt=None, image_data=None, **kwargs: (
        openai_complete_if_cache(
            "gpt-4o",
            prompt,
            system_prompt=system_prompt if system_prompt else None,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            history_messages=[
                (
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
                    }
                    if image_data
                    else {"role": "user", "content": prompt}
                ),
            ],
            **kwargs,
        )
        if image_data
        else openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=[],
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            **kwargs,
        )
    )

