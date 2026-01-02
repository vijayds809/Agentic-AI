import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from .agent_executor import RemittanceAgentExecutor


if __name__ == "__main__":

    executor = RemittanceAgentExecutor()

    remittance_skill = AgentSkill(
        id="remittance_status",
        name="Remittance Advice",
        description="Returns remittance advice or proof of payment",
        tags=["remittance", "payment", "finance"],
        examples=[
            "Provide remittance advice for payment 00104255",
            "Do you have remittance details for TAUK144863?"
        ],
    )

    card = AgentCard(
        name="Remittance Agent",
        description="Agent responsible for remittance-related queries",
        url="http://localhost:9002/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[remittance_skill],
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        ),
    )

    uvicorn.run(app.build(), host="0.0.0.0", port=9002)
