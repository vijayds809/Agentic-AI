import os, time, asyncio, json, re
from dotenv import load_dotenv

# Microsoft Agent Framework
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# OpenTelemetry + Azure Monitor
from opentelemetry import trace
from opentelemetry.metrics import get_meter_provider
from azure.monitor.opentelemetry import configure_azure_monitor

# Load env variables
load_dotenv()
configure_azure_monitor(connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING"))

# tracer
tracer = trace.get_tracer(__name__)

# Global token counters
GLOBAL_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

meter = get_meter_provider().get_meter("llm.metrics")
prompt_tokens_counter = meter.create_counter("msagent_llm_prompt_tokens", unit="tokens")
completion_tokens_counter = meter.create_counter("msagent_llm_completion_tokens", unit="tokens")
total_tokens_counter = meter.create_counter("msagent_llm_total_tokens", unit="tokens")
request_count_counter = meter.create_counter("msagent_llm_request_count", unit="count")



# -----------------------------------------------------
# NORMALIZATION FIX — robust conversion
# -----------------------------------------------------
def normalize_str(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return normalize_str(value[0]) if value else ""
    if isinstance(value, dict):
        return ""
    return str(value).strip()


# -----------------------------------------------------
# TOKEN METRICS COLLECTION
# -----------------------------------------------------
def record_llm_metrics(span, response, agent_name):
    usage = getattr(response, "usage", None)
    if not usage:
        span.set_attribute(f"{agent_name}.token_error", "usage missing")
        return

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", prompt + completion)

    span.set_attribute(f"{agent_name}.prompt_tokens", prompt)
    span.set_attribute(f"{agent_name}.completion_tokens", completion)
    span.set_attribute(f"{agent_name}.total_tokens", total)

    prompt_tokens_counter.add(prompt, {"agent": agent_name})
    completion_tokens_counter.add(completion, {"agent": agent_name})
    total_tokens_counter.add(total, {"agent": agent_name})

    GLOBAL_TOKEN_USAGE["prompt_tokens"] += prompt
    GLOBAL_TOKEN_USAGE["completion_tokens"] += completion
    GLOBAL_TOKEN_USAGE["total_tokens"] += total


# -----------------------------------------------------
# MOCK DATA
# -----------------------------------------------------
INVOICE_DATA = {
    "TAUK144863": {
        "Invoice Number": "TAUK144863",
        "Vendor Name": "Acme Supplies",
        "Status": "Paid",
        "Payment Date": "2025-10-31",
        "Net Amount": "1250.00",
        "Payment Reference": "00104255",
        "Due Date": "2025-10-30",
    },
    "HOLD999001": {
        "Invoice Number": "HOLD999001",
        "Vendor Name": "Globex Corp",
        "Status": "On Hold",
        "Hold Reason": "Missing PO approval",
        "Net Amount": "980.00",
        "Due Date": "2025-11-15",
    },
    "INV20251101": {
        "Invoice Number": "INV20251101",
        "Vendor Name": "Beta Industries",
        "Status": "Paid",
        "Payment Date": "2025-11-01",
        "Net Amount": "2100.00",
        "Payment Reference": "00223344",
        "Due Date": "2025-10-28",
    },
    "INV20251102": {
        "Invoice Number": "INV20251102",
        "Vendor Name": "Gamma Corp",
        "Status": "Paid",
        "Payment Date": "2025-11-02",
        "Net Amount": "3300.00",
        "Payment Reference": "00334455",
        "Due Date": "2025-10-29",
    },
}

REMITTANCE_DATA = {
    "00104255": [
        {
            "Vendor Name": "Acme Supplies",
            "Invoice Number": "TAUK144863",
            "Gross Amount": "1250.00",
            "Payment Reference": "00104255",
        }
    ],
    "00223344": [
        {
            "Vendor Name": "Beta Industries",
            "Invoice Number": "INV20251101",
            "Gross Amount": "2100.00",
            "Payment Reference": "00223344",
        }
    ],
    "00334455": [
        {
            "Vendor Name": "Gamma Corp",
            "Invoice Number": "INV20251102",
            "Gross Amount": "3300.00",
            "Payment Reference": "00334455",
        }
    ],
}



# -----------------------------------------------------
# TOOL WRAPPERS
# -----------------------------------------------------
async def tool_invoice_lookup(invoice_id):
    with tracer.start_as_current_span("tool.invoice_lookup"):
        return INVOICE_DATA.get(invoice_id.upper().strip(), {})


async def tool_remittance_lookup(reference):
    with tracer.start_as_current_span("tool.remittance_lookup"):
        ref = (reference or "").strip().upper()
        return REMITTANCE_DATA.get(ref, [])


# -----------------------------------------------------
# LLM CALL (LangGraph-style trace)
# -----------------------------------------------------
async def llm_call(client, prompt, agent_name):

    with tracer.start_as_current_span("call_model"):
        with tracer.start_as_current_span("runnablesequence"):

            with tracer.start_as_current_span("prompt") as p:
                p.set_attribute("prompt.text", (prompt or "")[:500])

            with tracer.start_as_current_span("azurechatopenai") as m:
                m.set_attribute("agent.name", agent_name)

                response = await client.client.chat.completions.create(
                    model=client.deployment_name,
                    messages=[{"role": "user", "content": prompt}]
                )

                record_llm_metrics(m, response, agent_name)
                return response


# -----------------------------------------------------
# INVOICE AGENT
# -----------------------------------------------------
class InvoiceAgent(ChatAgent):
    def __init__(self, client):
        super().__init__(chat_client=client, instructions="Invoice agent.")

    async def run(self, query):
        with tracer.start_as_current_span("agent.invoice_agent") as span:

            extraction = await llm_call(
                self.chat_client,
                f"Extract JSON keys: invoice_id, vendor_name, amount. Query: {query}",
                "agent.invoice_agent"
            )

            try:
                info = json.loads(extraction.choices[0].message.content)
            except:
                info = {}

            invoice_id = normalize_str(info.get("invoice_id"))

            if not invoice_id:
                # Try extracting from text
                match = re.search(r"\b[A-Z0-9]{6,}\b", query)
                invoice_id = match.group(0) if match else ""

            span.set_attribute("invoice_id.final", invoice_id)

            data = await tool_invoice_lookup(invoice_id)

            final = await llm_call(
                self.chat_client,
                f"Invoice data: {data}. Draft response.",
                "agent.invoice_agent"
            )

            return final.choices[0].message.content


# -----------------------------------------------------
# REMITTANCE AGENT
# -----------------------------------------------------
class RemittanceAgent(ChatAgent):
    def __init__(self, client):
        super().__init__(chat_client=client, instructions="Remittance agent.")

    async def run(self, query):
        with tracer.start_as_current_span("agent.remittance_agent") as span:

            extraction = await llm_call(
                self.chat_client,
                f"Extract payment_reference, vendor_name, amount as JSON. Query: {query}",
                "agent.remittance_agent"
            )

            try:
                info = json.loads(extraction.choices[0].message.content)
            except:
                info = {}

            # Save raw LLM output
            span.set_attribute("extraction.raw", extraction.choices[0].message.content)

            payment_reference = normalize_str(info.get("payment_reference"))

            # fallback: use pattern detection
            if not payment_reference:
                match = re.search(r"\b\d{8}\b", query)
                if match:
                    payment_reference = match.group(0)

            span.set_attribute("payment_reference.final", payment_reference)

            rem_data = await tool_remittance_lookup(payment_reference)

            final = await llm_call(
                self.chat_client,
                f"Remittance data: {rem_data}. Draft response.",
                "agent.remittance_agent"
            )

            return final.choices[0].message.content


# -----------------------------------------------------
# SUPERVISOR AGENT — root node
# -----------------------------------------------------
class SuperAgent(ChatAgent):
    def __init__(self, client, invoice_agent, remittance_agent):
        super().__init__(chat_client=client, instructions="Supervisor router.")
        self.invoice_agent = invoice_agent
        self.remittance_agent = remittance_agent

    async def run(self, query):
        with tracer.start_as_current_span("APPINSIGHTS_MSAGENTFRAMEWORK"):
            with tracer.start_as_current_span("Microsoft Agent Framework"):
                with tracer.start_as_current_span("agent.supervisor_agent"):
                    request_count_counter.add(1, {"agent": "supervisor_agent"})

                    # routing
                    routing = await llm_call(
                        self.chat_client,
                        f"Decide: invoice or remittance or both? Query: {query}",
                        "agent.supervisor_agent"
                    )

                    try:
                        route = routing.choices[0].message.content.lower()
                    except:
                        route = "both"

                    invoice_output = ""
                    rem_output = ""
                    extracted_ref = None

                    # Call invoice agent
                    if "invoice" in route or "both" in route:
                        invoice_output = await self.invoice_agent.run(query)

                        # Extract reference from invoice output
                        m = re.search(r"\b\d{8}\b", invoice_output)
                        if m:
                            extracted_ref = m.group(0)

                    # Prepare remittance query
                    if "remittance" in route or "both" in route:
                        rem_query = query
                        if extracted_ref:
                            rem_query += f"\nPayment Reference: {extracted_ref}"

                        rem_output = await self.remittance_agent.run(rem_query)

                    # Final combination
                    final = await llm_call(
                        self.chat_client,
                        f"Combine:\nInvoice: {invoice_output}\nRemittance: {rem_output}",
                        "agent.supervisor_agent"
                    )

                    return final.choices[0].message.content


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
async def main():

    client = AzureOpenAIChatClient(
        credential=AzureCliCredential(),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    )

    invoice_agent = InvoiceAgent(client)
    remittance_agent = RemittanceAgent(client)
    super_agent = SuperAgent(client, invoice_agent, remittance_agent)

    query = """
    Hi Team,
    Need invoice status for TAUK144863 and remittance for 00104255.
    """

    print("\n📩 Processing user query...\n")
    result = await super_agent.run(query)
    print(result)

    print("\n🔎 Total Token Usage:", GLOBAL_TOKEN_USAGE)


if __name__ == "__main__":
    asyncio.run(main())
