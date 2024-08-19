import os

from colorama import Fore, Style
from langchain_community.chat_models import GigaChat


class GigaChatProvider:

    def __init__(
        self,
        model,
        temperature,
        max_tokens,
        openai_api_key=None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = self.get_base_url()
        self.llm = self.get_llm_model()

    def get_base_url(self):
        """
        Gets the OpenAI Base URL from the environment variable if defined otherwise use the default one
        Returns:

        """
        base_url = os.environ.get("GIGACHAT_BASE_URL", None)
        return base_url


    def get_llm_model(self):
        # Initializing the chat model
        temperature = self.temperature
        if temperature <= 0:
            temperature = 0.01
        llm = GigaChat(
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens
        )
        if self.base_url:
            llm.base_url = self.base_url

        return llm

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            output = await self.llm.ainvoke(messages)

            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""

        return response
