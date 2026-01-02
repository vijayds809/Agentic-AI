from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError

from .agent import PaymentAgent


class PaymentAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = PaymentAgent()

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("Missing task_id or context_id")

        updater = TaskUpdater(queue, context.task_id, context.context_id)
        updater.start_work()

        try:
            result = await self.agent.process(context.get_user_input())
            await updater.add_artifact(
                [Part(root=TextPart(text=result))]
            )
            await updater.complete()

        except Exception as e:
            await updater.fail(str(e))

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        # Required by abstract base class
        raise ServerError(error=UnsupportedOperationError())
