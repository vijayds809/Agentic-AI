import re
from app.shared.tools import LocalTool, fetch_remittance
from app.shared.llm import call_llm

PROMPT = """
You are a remittance agent.

Rules:
- Use ONLY the TOOL_RESULT provided.
- If TOOL_RESULT is empty, say exactly:
  "No remittance data found for the given reference."
- If TOOL_RESULT contains records, present them in a clear table.
- Do NOT ask follow-up questions.
"""

class RemittanceAgent:
    def __init__(self):
        self.tool = LocalTool("remittance", fetch_remittance)

    async def process(self, query: str) -> str:
        # ✅ Explicitly extract payment_reference
        match = re.search(
            r"payment_reference\s*(?:is)?\s*(\d{6,12})",
            query,
            re.IGNORECASE
        )

        payment_ref = match.group(1) if match else None

        tool_result = await self.tool.ainvoke(
            {"PaymentReference": payment_ref} if payment_ref else {"query": query}
        )

        return await call_llm(
            PROMPT,
            f"TOOL_RESULT:\n{tool_result}\n\nQUERY:\n{query}"
        )
