from datetime import datetime
from .utils.views import print_agent_output
from .utils.llms import call_model
from langgraph.graph import StateGraph, END
import asyncio
import json

from ..memory.draft import DraftState
from . import ResearchAgent, ReviewerAgent, ReviserAgent


class EditorAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def plan_research(self, research_state: dict):
        """
        Curate relevant sources for a query
        :param summary_report:
        :return:
        :param total_sub_headers:
        :return:
        """

        initial_research = research_state.get("initial_research")
        task = research_state.get("task")
        max_sections = task.get("max_sections")
        include_human_feedback = task.get("include_human_feedback")
        human_feedback = research_state.get("human_feedback")

        prompt = [
            {
                "role": "system",
                "content": "Ты редактор исследования. Твоя цель — курировать исследовательский проект"
                " с самого начала и до завершения. Твоя основная задача — спланировать структуру"
                " разделов статьи на основе первоначального резюме исследования.\n ",
            },
            {
                "role": "user",
                "content": f"Сегодняшняя дата {datetime.now().strftime('%d/%m/%Y')}\n."
                f"Черновик исследовательского отчёта: '{initial_research}'\n"
                f"{f'Обратная связь от человека: {human_feedback}. Ты должен планировать разделы на основе обратной связи.' if include_human_feedback else ''}\n"
                f"Твоя задача — создать структуру заголовков разделов для исследовательского проекта"
                f" на основе приведённого выше резюме исследовательского отчёта.\n"
                f"Ты должен создать максимум {max_sections} заголовков разделов.\n"
                f"Ты должен сосредоточиться ТОЛЬКО на связанных с исследованием темах для подзаголовков и НЕ включать введение, заключение и ссылки.\n"
                f"ТЫ ДОЛЖЕН ОБЯЗАТЕЛЬНО вернуть JSON с полями 'title' (строка) и"
                f"'sections' (максимум {max_sections} заголовков разделов) со следующей структурой: "
                f"'{{title: название исследования, date: сегодняшняя дата, "
                f"sections: ['заголовок раздела 1', 'заголовок раздела 2', 'заголовок раздела 3' ...]}}.\n ",
            },
        ]

        print_agent_output(
            f"Planning an outline layout based on initial research...", agent="EDITOR"
        )
        response = await call_model(
            prompt=prompt,
            model=task.get("model"),
            response_format="json",
            api_key=self.headers.get("openai_api_key"),
        )
        plan = json.loads(response)

        return {
            "title": plan.get("title"),
            "date": plan.get("date"),
            "sections": plan.get("sections"),
        }

    async def run_parallel_research(self, research_state: dict):
        research_agent = ResearchAgent(self.websocket, self.stream_output, self.headers)
        reviewer_agent = ReviewerAgent(self.websocket, self.stream_output, self.headers)
        reviser_agent = ReviserAgent(self.websocket, self.stream_output, self.headers)
        queries = research_state.get("sections")
        title = research_state.get("title")
        workflow = StateGraph(DraftState)

        workflow.add_node("researcher", research_agent.run_depth_research)
        workflow.add_node("reviewer", reviewer_agent.run)
        workflow.add_node("reviser", reviser_agent.run)

        # set up edges researcher->reviewer->reviser->reviewer...
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "reviewer")
        workflow.add_edge("reviser", "reviewer")
        workflow.add_conditional_edges(
            "reviewer",
            (lambda draft: "accept" if draft["review"] is None else "revise"),
            {"accept": END, "revise": "reviser"},
        )

        chain = workflow.compile()

        # Execute the graph for each query in parallel
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "parallel_research",
                f"Running parallel research for the following queries: {queries}",
                self.websocket,
            )
        else:
            print_agent_output(
                f"Running the following research tasks in parallel: {queries}...",
                agent="EDITOR",
            )
        final_drafts = [
            chain.ainvoke(
                {
                    "task": research_state.get("task"),
                    "topic": query,
                    "title": title,
                    "headers": self.headers,
                }
            )
            for query in queries
        ]
        research_results = [
            result["draft"] for result in await asyncio.gather(*final_drafts)
        ]

        return {"research_data": research_results}
