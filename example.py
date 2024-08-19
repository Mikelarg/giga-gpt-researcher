from gpt_researcher import GPTResearcher
import asyncio
from dotenv import load_dotenv

from utils.enum import Tone

load_dotenv()


async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query=query, report_type=report_type, tone=Tone.Humorous)
    # Conduct research on the given query
    research_result = await researcher.conduct_research()
    # Write the report
    report = await researcher.write_report()
    return report


if __name__ == "__main__":
    query = "Сравнение Яндекс Станция 2 и SberBoom Home"
    report_type = "research_report"

    report = asyncio.run(get_report(query, report_type))
    print(report)
