"""
Simple AP Helpdesk Multi-Agent System with OpenInference Telemetry
Basic agent interactions with OpenInference observability and Azure Application Insights.

Azure Application Insights Setup:
1. Create an Application Insights resource in the Azure portal
2. Copy the connection string from the "Overview" page
3. Set the AZURE_APP_INSIGHTS_CONNECTION_STRING environment variable:
   - Windows: set         print("="*60)
        print("📊 TELEMETRY SUMMARY")
        print("="*60)
        print("✅ OpenInference traces captured for:")
        print("   • LLM calls and responses")
        print("   • Agent executions and handoffs") 
        print("   • Tool invocations")
        print("   • Workflow orchestration")
        print("   • Query processing and validation")
        print("🔍 Check console output above for detailed span information")
        
        # Azure Application Insights status
        if AZURE_APP_INSIGHTS_CONNECTION_STRING and not AZURE_APP_INSIGHTS_CONNECTION_STRING.startswith("InstrumentationKey=YOUR_"):
            print("☁️  Telemetry data sent to Azure Application Insights")
        else:
            print("📝 Azure Application Insights: Using placeholder connection string")
            print("   Set AZURE_APP_INSIGHTS_CONNECTION_STRING to enable cloud telemetry")
        
        print("="*60)NSIGHTS_CONNECTION_STRING="your_connection_string"
   - Linux/Mac: export AZURE_APP_INSIGHTS_CONNECTION_STRING="your_connection_string"
   - Or add it to your .env file
4. Run the application - telemetry will be sent to both console and Azure App Insights

Features:
- Multi-agent workflow orchestration (payment, remittance, validation, supervisor)
- OpenInference automatic LangChain instrumentation
- Custom workflow and agent spans with performance metrics
- Dual telemetry output: console (debugging) + Azure App Insights (production)
- Rich telemetry attributes: agent names, execution times, token usage, status
"""

import os
import asyncio
import time
import re
from dotenv import load_dotenv
import uuid
from collections import defaultdict

from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient

# === Configuration ===
# Azure Application Insights Configuration
# To enable Azure Application Insights telemetry:
# 1. Create an Application Insights resource in Azure
# 2. Get the connection string from the Azure portal
# 3. Set the AZURE_APP_INSIGHTS_CONNECTION_STRING environment variable OR
# 4. Replace the placeholder connection string below
# 
# Example connection string format:
# "InstrumentationKey=12345678-1234-1234-1234-123456789012;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/"

AZURE_APP_INSIGHTS_CONNECTION_STRING = os.getenv(
    "AZURE_APP_INSIGHTS_CONNECTION_STRING", 
    "InstrumentationKey=2c4c861b-348a-41c8-b217-ef2fa15e7843;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;ApplicationId=293f9b51-b3c7-45fe-bb74-a9a55eb75ab4"
)

# === OpenInference Telemetry Setup ===
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter

# Initialize OpenInference instrumentation
def setup_openinference_telemetry():
    """Setup OpenInference telemetry with Azure Application Insights"""
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": "ap-helpdesk-agents",
        "service.version": "1.0.0",
        "service.namespace": "langgraph",
        "deployment.environment": "development"
    })
    
    # Create tracer provider with resource
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Azure Application Insights exporter
    try:
        if AZURE_APP_INSIGHTS_CONNECTION_STRING and not AZURE_APP_INSIGHTS_CONNECTION_STRING.startswith("InstrumentationKey=YOUR_"):
            azure_exporter = AzureMonitorTraceExporter(
                connection_string=AZURE_APP_INSIGHTS_CONNECTION_STRING
            )
            azure_processor = BatchSpanProcessor(azure_exporter)
            tracer_provider.add_span_processor(azure_processor)
    except Exception as e:
        pass  # Silent fallback
    
    # Instrument LangChain
    LangChainInstrumentor().instrument()

# Environment Setup
load_dotenv()

# Initialize telemetry first
setup_openinference_telemetry()

# Get tracer for custom spans
tracer = trace.get_tracer(__name__)

# --- Metrics setup (mirror SK metrics) ---
try:
    APPINSIGHTS_CONNECTION_STRING = AZURE_APP_INSIGHTS_CONNECTION_STRING
    exporter_meter = AzureMonitorMetricExporter(connection_string=APPINSIGHTS_CONNECTION_STRING)
    reader = PeriodicExportingMetricReader(exporter_meter)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter_provider().get_meter("otel_ap_helpdesk")

    # Metrics (same names as SK for cross-framework comparison)
    request_count_counter = meter.create_counter(
        name="ap-helpdesk-agent_request-count",
        unit="count",
        description="Number of request per agent."
    )

    latency_histogram = meter.create_histogram(
        name="ap-helpdesk-agent_latency_ms-histogram",
        unit="seconds",
        description="Latency recorded by agent",
    )

    token_usage_counter = meter.create_counter(
        name="ap-helpdesk-token_usage-count",
        unit="tokens",
        description="Total number of tokens processed"
    )

    tool_latency = meter.create_histogram(
        name="ap-helpdesk-tool_latency_ms-histogram",
        unit="seconds",
        description="Latency of tool calls in seconds"
    )

    tool_usage_counter = meter.create_counter(
        name="ap-helpdesk-tool_usage-count",
        unit="count",
        description="Number of tool invocations by tool name"
    )

    tool_error_counter = meter.create_counter(
        name="ap-helpdesk-tool_errors-count",
        unit="count",
        description="Number of tool invocation errors by tool name"
    )

    tool_result_size_histogram = meter.create_histogram(
        name="ap-helpdesk-tool_result_size-histogram",
        unit="bytes",
        description="Size of tool result data"
    )

    validation_score_histogram = meter.create_histogram(
        name="ap-helpdesk-validation_score-histogram",
        unit="score",
        description="Quality validation scores (1-10)"
    )

    validation_status_counter = meter.create_counter(
        name="ap-helpdesk-validation_status-count",
        unit="count",
        description="Count of validation statuses (APPROVED/NEEDS_IMPROVEMENT)"
    )

    response_revision_counter = meter.create_counter(
        name="ap-helpdesk-response_revisions-count",
        unit="count",
        description="Number of times responses needed revision"
    )
except Exception:
    request_count_counter = None
    latency_histogram = None
    token_usage_counter = None
    tool_latency = None
    tool_usage_counter = None
    tool_error_counter = None
    tool_result_size_histogram = None
    validation_score_histogram = None
    validation_status_counter = None
    response_revision_counter = None
# --- end metrics setup ---

# Client/request globals (set in main per invocation)
CLIENT_NAME = None
REQUEST_ID = None

# In-memory accumulator to let supervisor aggregate tokens produced by child agents
TOKEN_ACCUM = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})

def _with_client_request(attrs: dict) -> dict:
    try:
        if CLIENT_NAME:
            attrs["client.name"] = CLIENT_NAME
        if REQUEST_ID:
            attrs["request.id"] = REQUEST_ID
    except Exception:
        pass
    return attrs

# === LLM Model Configuration ===
model = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    model_name="gpt-4o",
    azure_endpoint="https://apigatewayazeu.accenture.com/genai/stage/lbpass/",
    api_key="a7d9f5b4-d9ec-4397-b53c-08aebfebc7ae"
)

# === Tool Setup with Telemetry ===
async def get_tools(client):
    """Fetch available tools from MCP client with telemetry"""
    with tracer.start_as_current_span("mcp_tools_fetch") as span:
        tools = await client.get_tools()
        
        # Add telemetry attributes
        span.set_attribute("tools.count", len(tools))
        span.set_attribute("mcp.client_type", "MultiServerMCPClient")
        
        # Log tool names
        tool_names = [getattr(tool, 'name', 'unknown_tool') for tool in tools]
        span.set_attribute("tools.names", str(tool_names))
        
        return tools

# === Agent Instrumentation ===
def instrument_agent(agent, agent_name):
    """Add OpenInference telemetry to agent"""
    original_ainvoke = agent.ainvoke
    
    async def traced_ainvoke(*args, **kwargs):
        start = time.perf_counter()
        with tracer.start_as_current_span(f"agent.{agent_name}") as span:
            # Set agent attributes
            span.set_attribute("agent.name", agent_name)
            # Indicate this telemetry comes from the LangGraph implementation
            span.set_attribute("framework", "langraph")
            span.set_attribute("agent.type", "react_agent")
            
            try:
                # Capture input message count to isolate new messages for this invocation (avoid double counting tokens)
                try:
                    initial_messages_len = 0
                    if args:
                        first_arg = args[0]
                        if isinstance(first_arg, dict) and "messages" in first_arg and isinstance(first_arg["messages"], list):
                            initial_messages_len = len(first_arg["messages"])
                    elif "messages" in kwargs and isinstance(kwargs["messages"], list):
                        initial_messages_len = len(kwargs["messages"])
                except Exception:
                    initial_messages_len = 0

                result = await original_ainvoke(*args, **kwargs)

                # Record request count metric
                if request_count_counter:
                    request_count_counter.add(1, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph"}))

                # Record latency
                latency_s = time.perf_counter() - start
                span.set_attribute("agent.latency_s", latency_s)
                if latency_histogram:
                    latency_histogram.record(latency_s, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph"}))
                
                # Extract telemetry from result
                if isinstance(result, dict) and "messages" in result:
                    span.set_attribute("agent.messages_count", len(result["messages"]))
                    
                    # Get the last message for analysis
                    if result["messages"]:
                        last_message = result["messages"][-1]
                        content = getattr(last_message, 'content', '')
                        span.set_attribute("agent.response_length", len(content))
                        
                        # Check for tool calls and record tool metrics
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            span.set_attribute("agent.tool_calls_count", len(last_message.tool_calls))
                            for tc in last_message.tool_calls:
                                tool_name = getattr(tc, 'name', None) or getattr(tc, 'tool_name', 'unknown')
                                if tool_usage_counter:
                                    tool_usage_counter.add(1, attributes=_with_client_request({"tool.name": tool_name, "agent.name": agent_name, "framework": "langraph"}))
                                # Optionally record result size if present
                                try:
                                    res = getattr(tc, 'result', None)
                                    if res and tool_result_size_histogram:
                                        size = len(str(res).encode('utf-8'))
                                        tool_result_size_histogram.record(size, attributes=_with_client_request({"tool.name": tool_name, "agent.name": agent_name, "framework": "langraph"}))
                                except Exception:
                                    pass
                        # If this is a validation agent, try to parse a validation score and record metrics
                        try:
                            if "validation" in agent_name.lower():
                                import re as _re
                                score_match = _re.search(r"Validation Score:\s*(\d+)", content)
                                score = int(score_match.group(1)) if score_match else None
                                if score is not None and validation_score_histogram:
                                    validation_score_histogram.record(score, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph"}))
                                if score is not None and validation_status_counter:
                                    status = "APPROVED" if score >= 8 else "NEEDS_IMPROVEMENT"
                                    validation_status_counter.add(1, attributes=_with_client_request({"validation.status": status, "agent.name": agent_name, "framework": "langraph"}))
                                    if status == "NEEDS_IMPROVEMENT" and response_revision_counter:
                                        response_revision_counter.add(1, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph"}))
                        except Exception:
                            pass

                        # === Token Usage Recording (Actual from model) ===
                        try:
                            if token_usage_counter:
                                new_messages = []
                                try:
                                    # Determine slice of new messages produced in this invocation
                                    all_messages = result.get("messages", []) if isinstance(result, dict) else []
                                    if initial_messages_len < len(all_messages):
                                        new_messages = all_messages[initial_messages_len:]
                                except Exception:
                                    pass

                                total_input_tokens = 0
                                total_output_tokens = 0
                                total_tokens = 0

                                for m in new_messages:
                                    # LangChain AIMessage has usage_metadata dict with input_tokens, output_tokens, total_tokens
                                    usage_md = getattr(m, "usage_metadata", None)
                                    if isinstance(usage_md, dict):
                                        in_t = usage_md.get("input_tokens") or usage_md.get("prompt_tokens") or 0
                                        out_t = usage_md.get("output_tokens") or usage_md.get("completion_tokens") or 0
                                        tot_t = usage_md.get("total_tokens") or (in_t + out_t)
                                        total_input_tokens += in_t
                                        total_output_tokens += out_t
                                        total_tokens += tot_t

                                if total_tokens > 0:
                                    # Single counter; differentiate by token.type attribute
                                    token_usage_counter.add(total_input_tokens, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph", "token.type": "input_tokens"}))
                                    token_usage_counter.add(total_output_tokens, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph", "token.type": "output_tokens"}))
                                    token_usage_counter.add(total_tokens, attributes=_with_client_request({"agent.name": agent_name, "framework": "langraph", "token.type": "total_tokens"}))
                                    span.set_attribute("agent.token_usage_total", total_tokens)
                                    span.set_attribute("agent.token_usage_input", total_input_tokens)
                                    span.set_attribute("agent.token_usage_output", total_output_tokens)
                                    try:
                                        key = (REQUEST_ID or "unknown", agent_name)
                                        TOKEN_ACCUM[key]["input"] += total_input_tokens
                                        TOKEN_ACCUM[key]["output"] += total_output_tokens
                                        TOKEN_ACCUM[key]["total"] += total_tokens
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                
                span.set_attribute("agent.status", "success")
                return result
                
            except Exception as e:
                span.set_attribute("agent.status", "error")
                span.set_attribute("agent.error", str(e))
                raise
    
    agent.ainvoke = traced_ainvoke
    return agent

# === Agent Definitions ===

def create_payment_status_agent(model, tools):
    """Create the payment status agent"""
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=""" 
        You are a payment status agent. Assist in answering invoice/payment status related queries. You strictly DO NOT answer any remittance or other queries.
        - Strictly follow the below steps to answer invoice status, payment Status queries.

        Step 1. Information extraction: Extract the invoice details from the query, extract Invoice Number, Vendor Name, Date, Amount, Payment Reference(if available).
        Step 2. Fetch the invoice data using information extracted in Step 1 through tool call.
        Step 3. Check for the Status of invoice in the data fetched and create response as per below status based instructions. 
        -- Received - Check for Due Date and provide the update stating Invoice Received and expected due date for Payment as per the Due Date (if available).
        -- On Hold - (**Email need to drafted and sent only in case of any invoice with "On Hold" status)
            - Create email response to Invoice approval team, mention the invoice details and ask them approve the respective invoice. 
            - Send the drafted email via tool call. 
        -- Cancelled - Check for cancellation reason and draft response with the reason.
        -- Paid - Include Invoice Number, Payment Date, Net Amount and Payment Reference (if available) in your response.
        -- Partially Paid -  Include Invoice Number, Payment Date, Net Amount and Payment Reference (if available) in your response. Include the reason for Partial Payment and provide future payment date if any.
        Step 4. If no matching data found or if customer has not provided required inputs to fetch data, ask customer to provide a invoice copy. 
        Step 5. Capture all invoice data in a tabular format. DO NOT repeat the information in your response.
        """,
    name="payment_status_agent",
    )

def create_remittance_agent(model, tools):
    """Create the remittance advice agent"""
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=""" 
        You are a remittance agent. Assist in answering remittance related queries.
        - Strictly follow the below Steps to answer remittance related queries.

        Step 1. Information extraction: Extract Payment Amount, Payment Reference, vendor name (if available) from email for remittance request.
        Step 2. Fetch the remittance data using information extracted through tool call.
        Step 3. Include all records returned by Step 2 in your response. Response should include Vendor Name, Invoice Number, Gross Amount, Payment Reference fields. Capture the records in a tabular format. 
        Step 4. If no matching data found ask customer for additional information like cheque number or payment date. 

        - After you're done with your tasks, respond to the supervisor directly.
        - Respond ONLY with the results of your work, do NOT include ANY other text. 
        """,
    name="remittance_advise_agent",
    )

def create_validation_agent(model):
    """Create the validation agent"""
    return create_react_agent(
        model=model,
        tools=[],  # No tools needed for validation
        prompt=""" 
        You are a validation agent responsible for quality assurance of customer support responses.
        Your role is to evaluate whether the response provided by the support team adequately addresses the customer's original query.

        VALIDATION CRITERIA:
        1. COMPLETENESS: Does the response address all parts of the customer's query?
        2. ACCURACY: Is the information provided relevant and appropriate?
        3. CLARITY: Is the response clear and easy to understand?
        4. PROFESSIONALISM: Is the tone professional and helpful?
        5. COMPLETENESS OF DATA: If tabular data was requested, is it properly formatted and complete?

        VALIDATION PROCESS:
        Step 1. Analyze the original customer query to identify all requested information.
        Step 2. Review the provided response to check if it addresses each request.
        Step 3. Evaluate the quality and completeness of any data tables provided.
        Step 4. Check for any missing information or unclear statements.
        Step 5. Provide a validation score (1-10) and detailed feedback.

        OUTPUT FORMAT:
        - Validation Score: [1-10]
        - Query Coverage: [List what was requested vs what was provided]
        - Missing Elements: [Any gaps or missing information]
        - Quality Assessment: [Brief assessment of response quality]
        - Recommendations: [Suggestions for improvement if score < 8]
        - Overall Status: [APPROVED/NEEDS_IMPROVEMENT]

        Be thorough and objective in your evaluation.
        """,
    name="validation_agent",
    )

def create_supervisor_agent(model, agents):
    """Create the supervisor agent that coordinates other agents"""
    return create_supervisor(
        model=model,
        agents=agents,
        prompt="""
        - You are a Procure to Pay Finance and accounting support helpdesk expert who can resolve customer invoice and payment related queries. 
        - Your role is to ensure you answer all queries from the customer in the given email. 
        - Carefully read the email and understand the different query, and delegate the query to different expert agents to respond to customer.
        - You are managing three agents in assisting your tasks:
            - a payment_status_agent: Assign payment/invoice status related queries from the email to this agent.
            - a remittance_advise_agent: Assign remittance related queries from the email to this agent.
            - a validation_agent: Use this agent to validate the final response before sending to customer.
        
        WORKFLOW:
        1. Assign work to payment_status_agent and/or remittance_advise_agent as needed (one at a time, not in parallel).
        2. Collect responses from the specialized agents.
        3. Draft the final email response to customer based on the responses received.
        4. Send the original customer query AND your drafted response to validation_agent for quality check.
        5. If validation score is 8 or above, return the final response.
        6. If validation score is below 8, revise the response based on validation feedback and re-validate if needed.
        
        -Draft final email response to customer based on the response received from agents. DO NOT repeat the information in your response.
        -Include all invoice data in tabular format. DO NOT return intermediate output from agents.
        -Always validate your final response using the validation_agent before concluding.
        """,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

# === Main Execution with Telemetry ===
async def main():
    """Main function to run the multi-agent system with OpenInference telemetry"""
    
    with tracer.start_as_current_span("ap_helpdesk_workflow") as workflow_span:
        workflow_span.set_attribute("workflow.name", "AP Helpdesk Multi-Agent System")
        workflow_span.set_attribute("workflow.version", "1.0")
        
        # === MCP Client Setup ===
        with tracer.start_as_current_span("mcp_client_setup") as client_span:
            client = MultiServerMCPClient(
                {
                    "Invoice": {
                        "url": "https://iealite-ml-test.accenture.com/fastmcp/atcat/mcp/",
                        "transport": "streamable_http",
                        "headers": {
                            "X-Client-Auth": "a2e20f36140f8833c749274720dcfaaa917f1c50b3d99f4b1b650a0d5610f89e" 
                        }
                    }
                }
            )
            client_span.set_attribute("mcp.servers_count", 1)
            client_span.set_attribute("mcp.server_name", "Invoice")

        # === Get Tools ===
        tools = await get_tools(client)
        
        # === Create Agents with Telemetry ===
        with tracer.start_as_current_span("agents_creation") as agents_span:
            payment_status_agent = create_payment_status_agent(model, tools)
            payment_status_agent = instrument_agent(payment_status_agent, "payment_status_agent")

            remittance_advise_agent = create_remittance_agent(model, tools)
            remittance_advise_agent = instrument_agent(remittance_advise_agent, "remittance_advise_agent")

            validation_agent = create_validation_agent(model)
            validation_agent = instrument_agent(validation_agent, "validation_agent")

            supervisor_agent = create_supervisor_agent(
                model, 
                [payment_status_agent, remittance_advise_agent, validation_agent]
            )
            # instrument with canonical agent name 'supervisor_agent' to avoid duplicate metrics
            supervisor_agent = instrument_agent(supervisor_agent, "supervisor_agent")
            # Ensure the agent object's name matches the instrumented canonical name so any
            # internal references (or pretty_print) don't use a different name
            try:
                supervisor_agent.name = "supervisor_agent"
            except Exception:
                pass

            agents_span.set_attribute("agents.total_count", 4)
            agents_span.set_attribute("agents.specialized_count", 3)
            agents_span.set_attribute("agents.supervisor_count", 1)
        
        # Set client and request id for this invocation
        global CLIENT_NAME, REQUEST_ID
        CLIENT_NAME = "Client 2"
        REQUEST_ID = str(uuid.uuid4())

        # === Process User Query ===
        user_query = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Request-ID: {REQUEST_ID}
Client: {CLIENT_NAME}

Subject: Update on payment

Body:
Hi Team-
We need status of our invoice TAUK144863 which is due for payment. Additionally, we also need remittance details for the payment reference number:00104255, pls share both details.
Please advise.
Thanks
"""
                }
            ]
        }
        
        with tracer.start_as_current_span("query_processing") as query_span:
            query_content = user_query["messages"][0]["content"]
            query_span.set_attribute("query.length", len(query_content))
            query_span.set_attribute("query.contains_invoice", "TAUK144863" in query_content)
            query_span.set_attribute("query.contains_payment_ref", "00104255" in query_content)
            query_span.set_attribute("query.type", "combined_payment_remittance")
        
        # === Execute Workflow ===
        with tracer.start_as_current_span("supervisor_execution") as execution_span:
            try:
                # Supervisor-specific span to mirror SK supervisor instrumentation
                with tracer.start_as_current_span("supervisor.process") as sup_span:
                    sup_span.set_attribute("framework", "langraph")
                    sup_span.set_attribute("agent.name", "supervisor_agent")
                    # attach client/request context when available
                    try:
                        if CLIENT_NAME:
                            sup_span.set_attribute("client.name", CLIENT_NAME)
                        if REQUEST_ID:
                            sup_span.set_attribute("request.id", REQUEST_ID)
                    except Exception:
                        pass

                    sup_span.set_attribute("input.query_length", len(query_content))
                    sup_start = time.perf_counter()
                    messages = await supervisor_agent.ainvoke(user_query)

                    # Normalize any message objects produced by the supervisor so their
                    # .name reflects the canonical instrumented agent name 'supervisor_agent'.
                    # Some supervisor implementations may set message.name to 'supervisor'
                    # internally which can cause mixed labels in prints/metrics.
                    try:
                        if isinstance(messages, dict) and "messages" in messages:
                            for m in messages.get("messages", []):
                                try:
                                    if getattr(m, "name", None) == "supervisor":
                                        setattr(m, "name", "supervisor_agent")
                                except Exception:
                                    # best-effort normalization; continue on error
                                    pass
                    except Exception:
                        pass

                    # supervisor telemetry: latency + request count + subresponse count
                    sup_latency = time.perf_counter() - sup_start
                    sup_span.set_attribute("agent.latency_s", sup_latency)
                    if latency_histogram:
                        latency_histogram.record(sup_latency, attributes=_with_client_request({"agent.name": "supervisor_agent", "framework": "langraph"}))
                    # request_count for supervisor is recorded by the instrumented agent; avoid double-counting here

                    # subresponses count (approximate by messages length)
                    try:
                        sub_count = len(messages.get("messages", [])) if isinstance(messages, dict) else 0
                        sup_span.set_attribute("supervisor.subresponses_count", sub_count)
                    except Exception:
                        pass

                    # Supervisor aggregation: sum tokens recorded by agents for this request
                    try:
                        total_input = 0
                        total_output = 0
                        total_total = 0
                        for (req_id, a_name), vals in list(TOKEN_ACCUM.items()):
                            if req_id == (REQUEST_ID or "unknown"):
                                total_input += vals.get("input", 0)
                                total_output += vals.get("output", 0)
                                total_total += vals.get("total", 0)
                        if token_usage_counter and total_total > 0:
                            token_usage_counter.add(total_input, attributes=_with_client_request({"agent.name": "supervisor_agent", "framework": "langraph", "token.type": "input_tokens", "aggregation": "supervisor"}))
                            token_usage_counter.add(total_output, attributes=_with_client_request({"agent.name": "supervisor_agent", "framework": "langraph", "token.type": "output_tokens", "aggregation": "supervisor"}))
                            token_usage_counter.add(total_total, attributes=_with_client_request({"agent.name": "supervisor_agent", "framework": "langraph", "token.type": "total_tokens", "aggregation": "supervisor"}))
                            sup_span.set_attribute("supervisor.token_usage_total", total_total)
                            sup_span.set_attribute("supervisor.token_usage_input", total_input)
                            sup_span.set_attribute("supervisor.token_usage_output", total_output)
                    except Exception:
                        pass

                # Add execution telemetry
                execution_span.set_attribute("execution.status", "success")
                execution_span.set_attribute("execution.messages_count", len(messages["messages"]))

                # Analyze final response
                if messages["messages"]:
                    final_message = messages["messages"][-1]
                    final_content = getattr(final_message, 'content', '')
                    execution_span.set_attribute("execution.final_response_length", len(final_content))
                    execution_span.set_attribute("execution.contains_validation", "validation" in final_content.lower())

                workflow_span.set_attribute("workflow.status", "completed")

            except Exception as e:
                execution_span.set_attribute("execution.status", "error")
                execution_span.set_attribute("execution.error", str(e))
                workflow_span.set_attribute("workflow.status", "failed")
                raise
        
        # === Display Results ===
        print("\n" + "="*60)
        print("🎭 AP HELPDESK - FINAL RESPONSE")
        print("="*60)
        
        for i, message in enumerate(messages["messages"]):
            print(f"\n--- Message {i+1} ---")
            message.pretty_print()

        print("\n" + "="*60)
        
        # Azure Application Insights status
        if AZURE_APP_INSIGHTS_CONNECTION_STRING and not AZURE_APP_INSIGHTS_CONNECTION_STRING.startswith("InstrumentationKey=YOUR_"):
            print("☁️  Telemetry data sent to Azure Application Insights")
        else:
            print("📝 Azure Application Insights: Using placeholder connection string")
            print("   Set AZURE_APP_INSIGHTS_CONNECTION_STRING to enable cloud telemetry")
        
        print("="*60)

# === Entry Point ===
if __name__ == "__main__":
    asyncio.run(main())