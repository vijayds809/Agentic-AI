# remote_connections.py
"""
Handles:
- Discovering A2A remote agents
- Creating persistent A2AClient connections
- Low-level A2A send_message call wrapper (_call_agent_raw)
"""

import os
import uuid
import json
import logging
from dotenv import load_dotenv

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)

load_dotenv()
logger = logging.getLogger("remote_connection")

# Remote agent endpoints
REMOTE_AGENTS = {
    "Payment Agent": os.getenv("PAYMENT_AGENT_URL", "http://localhost:9001"),
    "Remittance Agent": os.getenv("REMITTANCE_AGENT_URL", "http://localhost:9002"),
}


# -------------------------------------------------------------------
# Discover agents and create persistent A2A clients
# -------------------------------------------------------------------
async def initialize_remote_connections() -> dict:
    connections = {}

    for name, addr in REMOTE_AGENTS.items():
        try:
            async with httpx.AsyncClient(timeout=30.0) as resolver_client:
                resolver = A2ACardResolver(resolver_client, addr)
                card = await resolver.get_agent_card()

            persistent_client = httpx.AsyncClient(timeout=30.0)

            conn = A2AClient(
                persistent_client,
                card,
                url=addr
            )

            connections[name] = conn
            logger.info("Discovered remote agent: %s @ %s", name, addr)

        except Exception as ex:
            logger.error(
                "Failed to discover agent %s at %s: %s",
                name,
                addr,
                ex,
            )

    return connections


# -------------------------------------------------------------------
# Low-level function to call a remote agent via A2A
# -------------------------------------------------------------------
async def call_agent_raw(agent_name: str, conn: A2AClient, query: str) -> list:
    """
    Sends query to a remote A2A agent and returns its 'parts' array.
    """
    message_id = str(uuid.uuid4())

    payload = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": query}],
            "messageId": message_id,
        },
        "meta": {}
    }

    params = MessageSendParams.model_validate(payload)
    request = SendMessageRequest(id=message_id, params=params)

    response: SendMessageResponse = await conn.send_message(request)

    if not isinstance(response.root, SendMessageSuccessResponse):
        logger.error("Agent %s returned non-success root: %r", agent_name, response.root)
        return []

    task: Task = response.root.result
    artifacts_json = json.loads(task.model_dump_json(exclude_none=True))

    parts = []
    for artifact in artifacts_json.get("artifacts", []):
        for p in artifact.get("parts", []):
            parts.append(p)

    return parts
