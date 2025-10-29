import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool

# Load Azure OpenAI credentials
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# 1. Azure OpenAI LLM setup
llm = LLM(
    model=deployment_name,       # Must match Azure deployment name
    base_url=endpoint,           # Azure endpoint
    api_key=api_key,
    api_version=api_version,     # Full API version string
    azure=True                   # Enable Azure compatibility
)

# 2. Scraping Tool
scrape_tool = ScrapeWebsiteTool()  # Instantiate without arguments!

# 3. Agent with scraping tool
search_agent = Agent(
    role="Web Scraper",
    goal="Scrape necessary information from the provided webpage to answer any user query.",
    backstory=(
        "You are a research specialist who uses the ScrapeWebsiteTool to extract data from the internet."
    ),
    tools=[scrape_tool],   # Assign the tool
    verbose=True,
    allow_delegation= False,
    llm=llm                # Assign Azure LLM
)

# 4. Query and URL to scrape
customer_query = "How do I add memory to a CrewAI Crew?"
webpage_url = "https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"

# 5. Task for agent
search_task = Task(
    description=f"""The customer has asked: "{customer_query}"
    Scrape the documentation page: {webpage_url}
    Use the information you extract to provide a direct, complete, step-by-step answer.""",
    expected_output="A clear, accurate answer to the customer's memory question, with reference to the documentation.",
    agent=search_agent,
)

# 6. (Optional) Review agent and review task
review_agent = Agent(
    role="Response Reviewer",
    goal="Ensure the answer is complete, accurate, and easy for the customer to understand.",
    backstory="You improve the answers for customer-friendliness and clarity.",
    verbose=True,
    llm=llm
)

review_task = Task(
    description="Review and refine the research agent’s answer for final delivery.",
    expected_output="A polished, customer-ready response.",
    agent=review_agent,
)

# 7. Create and run the Crew
crew = Crew(
    agents=[search_agent, review_agent],
    tasks=[search_task, review_task],
    verbose=True,
    memory=True
)

result = crew.kickoff()
print("FINAL ANSWER:\n", result)
