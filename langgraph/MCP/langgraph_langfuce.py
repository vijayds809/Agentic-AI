import os
from dotenv import load_dotenv
import asyncio
import time
from functools import wraps
from collections import defaultdict

from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient

# from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter
from langchain.tools import Tool

from langfuse import get_client
#from langchain.callbacks import LangfuseCallbackHandler
from langfuse.langchain import CallbackHandler


# === ENV + Tracing Setup ===
load_dotenv()

# langfuse = Langfuse(
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#     host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")  # Optional
# )
langfuse = get_client()

langfuse_handler = CallbackHandler()

# === LLM Model ===
model = AzureChatOpenAI(
    # Fill in your model configuration here
    azure_deployment="gpt-4o",
    api_version = "2024-08-01-preview",
    model_name="gpt-4o",
    azure_endpoint="https://apigatewayazeu.accenture.com/genai/stage/lbpass/",
api_key = "a7d9f5b4-d9ec-4397-b53c-08aebfebc7ae",
callbacks=[langfuse_handler]
)

# === Metrics Tracking ===
request_counts = defaultdict(int)

# === Instrumentation Wrapper for Agents ===

# === Tool Invocation Wrapper (example for get_tools) ===
async def get_tools_with_telemetry(client):    
        tools = await client.get_tools()
        return tools                

# === Main Execution ===
async def main():

    # trace = langfuse.trace(
    # name="AH-usecase-Agent",
    # metadata={
    #     "environment": "development",
    #     "version": "1.0"
    # }
    # )
    # === MCP Client Setup ===
    client = MultiServerMCPClient(
        {
            "Invoice":{
                    "url":"https://iealite-ml-test.accenture.com/fastmcp/atcat/mcp/",
                    "transport": "streamable_http",
                    "headers": {
                        "X-Client-Auth": "a2e20f36140f8833c749274720dcfaaa917f1c50b3d99f4b1b650a0d5610f89e" 
                    }
                }
        }
    )

    tools = await get_tools_with_telemetry(client)
    
    # === Define & Instrument Agents ===
    payment_status_agent = create_react_agent(
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

    remittance_advise_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=""" 
        You are a remittance agent. Assist in answering remittance related queries.
        - Strictly follow the below Steps to answer remittance related queries.

        Step 1. Information extraction: Extract Payment Amount, Payment Reference, vendor name (if available) from email for remittance request.
        Step 2. Fetch the remittance data using information extracted through tool call.
        Step 3. Incldue all records returned by Step 2 in your response. Response should include Vendor Name, Invoice Number, Gross Amount, Payment Reference fields. Capture the records in a tabular format. 
        Step 4. If no matching data found ask customer for addional information like cheque number or payment date. 

        - After you're done with your tasks, respond to the supervisor directly.
        - Respond ONLY with the results of your work, do NOT include ANY other text. 
        """,
        name="remittance_advise_agent",
    )
    # Instrument agent methods
    #payment_status_agent = instrument_agent(payment_status_agent)
    #remittance_advise_agent = instrument_agent(remittance_advise_agent)

    # === Supervisor ===
    supervisor = create_supervisor(
        model=model,
        agents=[payment_status_agent, remittance_advise_agent],
        prompt="""
        - You are a Procure to Pay Finance and accounting support helpdesk expert who can resolve customer invoice and payment related queries. 
        - Your role is to ensure you answer all queries from the customer in the given email. 
        - Carefully read the email and understand the different query, and delegate the query to differnt expert agents to respond to customer.
        - You are managing two agents in assisting your tasks.
            - a payment_status_agent. Assign payment/invoice status related queries from the email to this agent.
            - a remittance_advise_agent. Assign remittance related queries from the email to this agent.
        -Assign work to one agent at a time, do not call agents in parallel.
        -Draft final email response to customer based on the response received from agents. DO NOT repeat the information in your response.
        -RETURN only the final email response. Include all invoice data in tabular format. DO NOT return intermediate output from agents.

        """,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

    # === User Query ===
    user_query = {
        "messages": [
            {
                "role": "user",
                "content": """
                Subject :  Update on payment

                Body:
                Hi Team-
                We need status of our invoice TAUK144863 which is due for payment. Additionally, we also need remittance details for the payment reference number:00104255, pls share both details.    
                Please advise.
                Thanks
                """
            }
        ]
    }

    # === Supervisor Invocation with Span ===
    #try:
    #messages = await supervisor.ainvoke(user_query, config={"callbacks": [langfuse_handler]})
    
    messages = await supervisor.ainvoke(user_query, config={"callbacks": [langfuse_handler], "metadata": {"langfuse_tags": ["laghgraph-APH-Usecase"]}})
    
    #messages = await supervisor.ainvoke(user_query, config={"callbacks": [langfuse_handler], "metadata": {"name": "laghgraph-APH-Usecase"}})    
    
    for m in messages["messages"]:
        m.pretty_print()

    #     trace.generation(
    #         name="final_response",
    #         input=user_query["messages"][-1]["content"],
    #         output=messages["messages"][-1].content if messages["messages"] else "",
    #     )

    # finally:
    #     # End the trace
    #     trace.end()
    #     # Ensure all events are sent
    #     langfuse.flush()

if __name__ == "__main__":
    asyncio.run(main())
