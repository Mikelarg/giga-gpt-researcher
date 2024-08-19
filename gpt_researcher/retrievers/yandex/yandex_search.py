# Tavily API Retriever

# libraries
import os
from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper
from duckduckgo_search import DDGS


class YandexSearch():
    """
    Tavily API Retriever
    """
    def __init__(self, query, headers=None, topic="general"):
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.query = query
        self.headers = headers or {}
        self.api_key = self.get_api_key()
        self.foder_id = self.get_folder_id()
        self.client = YandexSearchAPIWrapper(
            api_key=self.api_key, folder_id=self.foder_id,
            answer_fields=["url", "content", "title"]
        )
        self.topic = topic

    def get_api_key(self):
        """
        Gets the Yandex API key
        Returns:

        """
        api_key = self.headers.get("yandex_api_key")
        if not api_key:
            try:
                api_key = os.environ["YANDEX_API_KEY"]
            except KeyError:
                raise Exception("Yandex API key not found. Please set the YANDEX_API_KEY environment variable.")
        return api_key

    def get_folder_id(self):
        """
        Gets the Yandex Folder id
        Returns:

        """
        folder_id = self.headers.get("yandex_folder_id")
        if not folder_id:
            try:
                folder_id = os.environ["YANDEX_FOLDER_ID"]
            except KeyError:
                raise Exception("Yandex Folder id not found. Please set the YANDEX_FOLDER_ID environment variable.")
        return folder_id

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        try:
            # Search the query
            results = self.client.results(self.query)
            # Return the results
            search_response = [
                {
                    "href": obj["url"],
                    "body": obj["content"],
                    "title": obj["title"]
                } for obj in results
            ]
        except Exception as e: # Fallback in case overload on Tavily Search API
            print(f"Error: {e}. Fallback to DuckDuckGo Search API...")
            try:
                ddg = DDGS()
                search_response = ddg.text(self.query, region='wt-wt', max_results=max_results)
            except Exception as e:
                print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
                search_response = []
        return search_response
