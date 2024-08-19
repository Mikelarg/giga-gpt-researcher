from datetime import datetime
import json5 as json
from .utils.views import print_agent_output
from .utils.llms import call_model

sample_json = """
{
  "table_of_contents": Оглавление в синтаксисе Markdown (используя '-') на основе ВСЕХ заголовков и ВСЕХ подзаголовков исследования,
  "introduction": Подробное введение в тему в синтаксисе Markdown с гиперссылками на соответствующие источники,
  "conclusion": Заключение ко всему исследованию на основе всех данных исследования в синтаксисе Markdown с гиперссылками на соответствующие источники,
  "sources": Список строк со всеми использованными ссылками на источники во всех данных исследования в синтаксисе Markdown и формате цитирования APA. Пример: ['-  Название, год, Автор [url источника](url источника)', ...]
}
"""


class WriterAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers

    def get_headers(self, research_state: dict):
        return {
            "title": research_state.get("title"),
            "date": "Дата",
            "introduction": "Введение",
            "table_of_contents": "Оглавление",
            "conclusion": "Заключение",
            "references": "Источники",
        }

    async def write_sections(self, research_state: dict):
        query = research_state.get("title")
        data = research_state.get("research_data")
        task = research_state.get("task")
        follow_guidelines = task.get("follow_guidelines")
        guidelines = task.get("guidelines")

        data_inline = "\n".join(
            [f"{key}: {value}" for topic in data for key, value in topic.items()]
        )

        prompt = [
            {
                "role": "system",
                "content": "Ты научный писатель. Твоя цель — писать хорошо составленные "
                "исследовательские отчёты по "
                "теме на основе результатов и информации исследований.\n ",
            },
            {
                "role": "user",
                "content": f"Сегодняшняя дата {datetime.now().strftime('%d/%m/%Y')}\n."
                f"Запрос или тема: {query}\n"
                f"Данные исследования: {data_inline}\n"
                f"Твоя задача — написать глубокое, хорошо составленное и подробное "
                f"введение и заключение к исследовательскому отчёту на основе предоставленных данных исследования. "
                f"Не включай заголовки в результаты.\n"
                f"ТЫ ДОЛЖЕН ОБЯЗАТЕЛЬНО включить все соответствующие источники в введение и заключение в виде гиперссылок в синтаксисе markdown —"
                f"Например: 'Это пример текста. ([url website](url))'\n\n"
                f"{f'Ты должен следовать предоставленным рекомендациям: {guidelines}' if follow_guidelines else ''}\n"
                f"ТЫ ДОЛЖЕН ОБЯЗАТЕЛЬНО вернуть JSON в следующем формате:\n"
                f"{sample_json}\n\n",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            max_retries=2,
            response_format="json",
            api_key=self.headers.get("openai_api_key"),
        )
        return json.loads(response)

    async def revise_headers(self, task: dict, headers: dict):
        prompt = [
            {
                "role": "system",
                "content": """Ты научный писатель. 
Твоя единственная цель — пересмотреть данные заголовков на основе предоставленных рекомендаций.""",
            },
            {
                "role": "user",
                "content": f"""Твоя задача — пересмотреть предоставленный JSON с заголовками на основе данных рекомендаций.
Ты должен следовать рекомендациям, но значения должны быть простыми строками, без использования синтаксиса markdown.
ТЫ ДОЛЖЕН ОБЯЗАТЕЛЬНО вернуть JSON в том же формате, что и в данных заголовков.
Рекомендации: {task.get("guidelines")}\n
Данные заголовков: {headers}\n
""",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
            api_key=self.headers.get("openai_api_key"),
        )
        return {"headers": json.loads(response)}

    async def run(self, research_state: dict):
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "writing_report",
                f"Writing final research report based on research data...",
                self.websocket,
            )
        else:
            print_agent_output(
                f"Writing final research report based on research data...",
                agent="WRITER",
            )

        research_layout_content = await self.write_sections(research_state)

        if research_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                research_layout_content_str = json.dumps(
                    research_layout_content, indent=2
                )
                await self.stream_output(
                    "logs",
                    "research_layout_content",
                    research_layout_content_str,
                    self.websocket,
                )
            else:
                print_agent_output(research_layout_content, agent="WRITER")

        headers = self.get_headers(research_state)
        if research_state.get("task").get("follow_guidelines"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "rewriting_layout",
                    "Rewriting layout based on guidelines...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    "Rewriting layout based on guidelines...", agent="WRITER"
                )
            headers = await self.revise_headers(
                task=research_state.get("task"), headers=headers
            )
            headers = headers.get("headers")

        return {**research_layout_content, "headers": headers}
