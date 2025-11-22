import os 
from crewai import LLM,Agent,Task,Crew 
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()

search_tool = SerperDevTool()
llm = LLM(
    model = 'huggingface/meta-llama/Llama-3.1-8B-Instruct',
    api_key = os.getenv("HF_Token"),
    # max_tokens = 1024,
    # max_input_tokens = 4000,
    # stream = False
)

search_agent = Agent(
    role = "Web Scraper",
    goal = "you need to scrape the web efficiently and get the content belongs to the given topic.",
    backstory = (
        "as a Web Scraper, you are a master in searching the web "
        "and get the accurate information needed."
    ),
    llm = llm,
    tools = [search_tool],
    verbose = True
)

content_generator = Agent(
    role = "Social Media Content Writer",
    goal = "generate the content that can be used to post in social media",
    backstory = (
        "you are a specialist in generating the content to post in social media."
        "you have the capability to crafts the posts that fits well to post on social media."
    ),
    llm = llm
)

search_agent_task = Task(
    description=(
        "scrape the web and generate the  content based on the {topic}"
    ),
    expected_output = (
        "a clean,fact-based content on the {topic} that"
        "can be used to craft social media posts content."
    ),
    agent = search_agent
)

content_generator_task = Task(
    description=(
        "you need to create posts for the following social media apps."
        "twitter,instagram and whatsapp stauts with appropiate hashtags."
    ),
    expected_output=(
        "a clean and controversial-free social media posts with appropiate posts and atleast 3-4 lines in lenght "
        "in json format like key is platform name and value is content that can be used for posting."
    ),
    agent = content_generator
)

crew = Crew(
    agents = [search_agent,content_generator],
    tasks = [search_agent_task,content_generator_task],
    verbose = True
)

rel = crew.kickoff(
    inputs = {
        'topic' : "International Men's Day"
    }
)