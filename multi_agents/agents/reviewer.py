from .utils.views import print_agent_output
from .utils.llms import call_model

TEMPLATE = """Ты эксперт по рецензированию научных статей. \
Твоя цель — рецензировать черновики исследований и предоставлять обратную связь автору, основываясь только на конкретных рекомендациях. \
"""


class ReviewerAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def review_draft(self, draft_state: dict):
        """
        Review a draft article
        :param draft_state:
        :return:
        """
        task = draft_state.get("task")
        guidelines = "- ".join(guideline for guideline in task.get("guidelines"))
        revision_notes = draft_state.get("revision_notes")

        revise_prompt = f"""Автор уже внес изменения в черновик на основе твоих предыдущих замечаний с следующими комментариями:
{revision_notes}\n
Пожалуйста, предоставь дополнительную обратную связь ТОЛЬКО в случае критической необходимости, так как автор уже внёс изменения на основе ваших предыдущих комментариев.
Если ты считаешь, что статья достаточна или что требуются незначительные правки, ответь "None".
"""

        review_prompt = f"""Тебе поручено рецензировать черновик, который был написан не экспертом на основе конкретных рекомендаций.
Пожалуйста, прими черновик, если он достаточно хорош для публикации, или отправьте его на доработку вместе с твоими замечаниями, чтобы направить процесс доработки.
Если не все критерии рекомендаций выполнены, ты должен предоставить соответствующие комментарии для доработки.
Если черновик соответствует всем рекомендациям, пожалуйста, ответь "None".
{revise_prompt if revision_notes else ""}

Рекомендации: {guidelines}\nЧерновик: {draft_state.get("draft")}\n
"""
        prompt = [
            {"role": "system", "content": TEMPLATE},
            {"role": "user", "content": review_prompt},
        ]

        response = await call_model(
            prompt, model=task.get("model"), api_key=self.headers.get("openai_api_key")
        )

        if task.get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "review_feedback",
                    f"Review feedback is: {response}...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Review feedback is: {response}...", agent="REVIEWER"
                )

        if "None" in response:
            return None
        return response

    async def run(self, draft_state: dict):
        task = draft_state.get("task")
        guidelines = task.get("guidelines")
        to_follow_guidelines = task.get("follow_guidelines")
        review = None
        if to_follow_guidelines:
            print_agent_output(f"Reviewing draft...", agent="REVIEWER")

            if task.get("verbose"):
                print_agent_output(
                    f"Following guidelines {guidelines}...", agent="REVIEWER"
                )

            review = await self.review_draft(draft_state)
        else:
            print_agent_output(f"Ignoring guidelines...", agent="REVIEWER")
        return {"review": review}
