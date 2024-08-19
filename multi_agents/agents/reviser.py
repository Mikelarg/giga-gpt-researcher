from .utils.views import print_agent_output
from .utils.llms import call_model
import json

sample_revision_notes = """
{
  "draft": { 
    название черновика: Пересмотренный черновик, который ты отправляешь на рецензирование 
  },
  "revision_notes": Твое сообщение рецензенту о внесённых изменениях в черновик на основе его комментариев
}
"""


class ReviserAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def revise_draft(self, draft_state: dict):
        """
        Review a draft article
        :param draft_state:
        :return:
        """
        review = draft_state.get("review")
        task = draft_state.get("task")
        draft_report = draft_state.get("draft")
        prompt = [
            {
                "role": "system",
                "content": "Ты эксперт в написании текстов. Твоя задача — доработать черновики на основе замечаний рецензента.",
            },
            {
                "role": "user",
                "content": f"""Черновик:\n{draft_report}" + "Замечания рецензента:\n{review}\n\n
Твой рецензент поручил вам доработать следующий черновик, который был написан не экспертом.
Если ты решишь следовать замечаниям рецензента, пожалуйста, напиши новый черновик и убедись, что ты учел все поднятые им вопросы.
Пожалуйста, оставь все остальные аспекты черновика без изменений.
ТЫ ДОЛЖЕН ОБЯЗАТЕЛЬНО вернуть JSON в следующем формате:
{sample_revision_notes}
""",
            },
        ]

        response = await call_model(
            prompt,
            model=task.get("model"),
            response_format="json",
            api_key=self.headers.get("openai_api_key"),
        )
        return json.loads(response)

    async def run(self, draft_state: dict):
        print_agent_output(f"Rewriting draft based on feedback...", agent="REVISOR")
        revision = await self.revise_draft(draft_state)

        if draft_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "revision_notes",
                    f"Revision notes: {revision.get('revision_notes')}",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Revision notes: {revision.get('revision_notes')}", agent="REVISOR"
                )

        return {
            "draft": revision.get("draft"),
            "revision_notes": revision.get("revision_notes"),
        }
