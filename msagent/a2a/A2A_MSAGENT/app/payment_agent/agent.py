import re
from app.shared.tools import LocalTool, fetch_invoice_status
from app.shared.llm import call_llm

PROMPT = """
You are a payment status agent.
Only answer payment / invoice status questions.
"""

class PaymentAgent:
    def __init__(self):
        self.tool = LocalTool("payment", fetch_invoice_status)

    async def process(self, query: str) -> str:
        inv = re.search(r"TAUK\d+", query)
        tool_result = await self.tool.ainvoke(
            {"InvoiceNumber": inv.group()} if inv else {"query": query}
        )

        return await call_llm(
            PROMPT,
            f"TOOL_RESULT:\n{tool_result}\nQUERY:\n{query}"
        )
