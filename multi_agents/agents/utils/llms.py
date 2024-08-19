import os

from langchain_community.chat_models.gigachat import _convert_dict_to_message
from langchain_community.chat_models import GigaChat
from langchain_community.adapters.openai import convert_openai_messages
from gigachat.models import Messages
from langchain_openai import ChatOpenAI

from gpt_researcher import Config


async def call_model(
    prompt: list,
    model: str,
    max_retries: int = 2,
    response_format: str = None,
    api_key: str = None,
) -> str:
    cfg = Config()
    if cfg.llm_provider == "openai":
        optional_params = {}
        if response_format == "json":
            optional_params = {"response_format": {"type": "json_object"}}
        lc_messages = convert_openai_messages(prompt)
        response = (
            ChatOpenAI(
                model=model,
                max_retries=max_retries,
                model_kwargs=optional_params,
                api_key=api_key,
            )
            .invoke(lc_messages)
            .content
        )
        return response
    elif cfg.llm_provider == "gigachat":
        lc_messages = []
        for m in prompt:
            lc_messages.append(_convert_dict_to_message(Messages.parse_obj(m)))
        response = (
            GigaChat(model=model, max_retries=max_retries).invoke(lc_messages).content
        )
        return response
    else:
        raise Exception(f"LLM Provider {cfg.llm_provider} not supported")
