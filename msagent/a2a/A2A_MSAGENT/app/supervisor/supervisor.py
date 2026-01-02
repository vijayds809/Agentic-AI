import asyncio
import json
import logging

from app.shared.llm import call_llm
from .remote_connections import initialize_remote_connections, call_agent_raw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supervisor")

ROUTER_PROMPT = """
You are a routing agent.

Return ONLY valid JSON.
Do NOT add explanations.

Schema:
{
  "agents": ["Payment Agent", "Remittance Agent"]
}

Rules:
- If query asks about invoice/payment status → Payment Agent
- If query asks about remittance/proof of payment → Remittance Agent
"""

async def main():
    # 1. Discover agents
    connections = await initialize_remote_connections()

    if not connections:
        logger.error("No agents discovered. Start agents first.")
        return

    user_query = (
        "We need status of invoice TAUK144863 and "
        "remittance details for payment_reference 00104255"
    )

    # 2. Ask LLM for routing decision
    raw = await call_llm(ROUTER_PROMPT, user_query)

    try:
        routing = json.loads(raw)
        agents = routing.get("agents", [])
    except Exception as e:
        logger.error("Invalid router output: %s", raw)
        return

    # 3. Call selected agents
    for agent_name in agents:
        if agent_name not in connections:
            logger.warning("Agent not available: %s", agent_name)
            continue

        parts = await call_agent_raw(
            agent_name,
            connections[agent_name],
            user_query
        )

        print(f"\n--- {agent_name} ---")
        for p in parts:
            print(p.get("text", ""))

if __name__ == "__main__":
    asyncio.run(main())
