import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from .agent_executor import PaymentAgentExecutor


if __name__ == "__main__":

    executor = PaymentAgentExecutor()

    payment_skill = AgentSkill(
        id="payment_status",
        name="Payment Status",
        description="Returns payment status for a given invoice",
        tags=["payment", "invoice", "finance"],
        examples=[
            "What is the payment status of TAUK144863?",
            "Has invoice TAUK144863 been paid?"
        ],
    )

    card = AgentCard(
        name="Payment Agent",
        description="Agent responsible for invoice and payment status queries",
        url="http://localhost:9001/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[payment_skill],
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        ),
    )

    uvicorn.run(app.build(), host="0.0.0.0", port=9001)
