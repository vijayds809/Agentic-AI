import os 
from dotenv import load_dotenv 
load_dotenv()
from crewai import LLM, Agent, Task, Crew 
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool


# google_search_tool = SerperDevTool()
# url_scrape_tool = ScrapeWebsiteTool()
#rag_over_url_tool = WebsiteSearchTool()

one_page_scrape_tool = ScrapeWebsiteTool(
    website_url = "https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
base_url = os.getenv("AZURE_OPENAI_ENDPOINT")  # Typically https://<resource>.openai.azure.com/
api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # e.g., "2024-08-01-preview"
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # e.g., gpt-4o or your deployment name

# Note (critical!): "model" must be your Azure deployment name, not a generic like "gpt-4.1-global"
llm = LLM(
    model=deployment_name,      # Always your deployment name, matches what Azure shows for "deployment"
    base_url=base_url,          # Not "endpoint"
    api_key=api_key,            # Not "key"
    api_version=api_version,    # Must be full version string, e.g. "2024-08-01-preview"
    azure=True                  # Forces Azure compatibility in CrewAI/LiteLLM
)

support_agent = Agent(
    role = "Senior Support Representative",
    goal = """
        Be the most friendly and helpful support representative in your team.
        """,
    backstory = """"
        you work at crewai(https://crewai.com) as a senior customer support representative
        and , you are going to support the {customer}, a super important customer of the company
        you need to make sure that, you are going to provide him best support experience
        make sure you provide him all the answers without any assumptions.
        """,
    allow_delegation = False,
    verbose = True
)

support_quality_assurance_agent = Agent(
    role = "Support Quality Assurance Agent",
    goal = """
        Get recognition by providing the best quality assurance for customer support interactions
        in your team.
        """,
    backstory = """
        you work at crewai(https://crewai.com) as a support quality assurance agent
        your job is to make sure that the support interactions provided by the support agents
        meet the highest quality standards, complete answers and make no assumptions.
        """,
    verbose = True
)


inquiry_response_task = Task(
    description="""
        {customer} has reached out with the following inquiry:
        {inquiry}
        {person} from the {customer} is the one who reached out.
        strive hard to provide him with best possible support experience and
        you must strive hard to provide complete and accurate respomnses without making any assumptions.
        """,
    expected_output = """
        a detailed, informative response to the customer's inquiry,
        the response should include all the references and sources
        that you used to gather the information, including
        links to relevant documentation, articles, or other resources.
        ensure the answer is complete and accurate and
        leaving no questions unanswered and maintaining
        a friendly and helpful tone throughout.
        """,
    agent = support_agent,
    tools = [one_page_scrape_tool],
)

quality_assurance_task = Task(
    description = """
        review the response provided by the senior support representative
        ensure that the response is complete, accurate, and helpful.
        make sure that the response addresses all aspects of the customer's inquiry
        inspect the tools used to gather information and ensure they are appropriate and effective.
        """,
    expected_output = """
        a final,detailed and informative response 
        ready to send to the customer.
        the response should address all the customer's inquiries,incorporating 
        all relevant feedback and improvements
        don't be formal as we are chill and cool company
        but, maintain a professional tone while being friendly and approachable throughout.
        """,
    agent = support_quality_assurance_agent
)

crew = Crew(
    agents = [support_agent, support_quality_assurance_agent],
    tasks = [inquiry_response_task, quality_assurance_task],
    verbose = True,
    memory = True
)

customer = "DeepLearningAI"
person = "Andrew NG"
inquiry = """
    I need help in setting up the crew
    and kicking it off especially
    how can i add memory to the crew?
    can you provide guidance?"""


rel = crew.kickoff(inputs={
    "customer": customer,
    "person": person,
    "inquiry": inquiry,
})


print(rel)

