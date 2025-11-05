from crewai import LLM,Agent,Task,Crew 
from crewai_tools import SerperDevTool 
from dotenv import load_dotenv 
load_dotenv()

import os 

search_tool = SerperDevTool()

llm = LLM(
    model = "google/models/gemini-2.5-flash",
    api_key = os.getenv("Google_API_Key"),
    verbose = True
)

llm_2 = LLM(
    model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",     # 👈 Open-source HF model    # 👈 Hugging Face OpenAI-compatible router
    api_key=os.getenv("HF_Token"),               # 👈 Your Hugging Face access token
    verbose=True
)

search_agent = Agent(
    role = "Google Search Agent",
    goal = (
        "search google for the {topic} and return the content that"
        "will helpful to give response to the user in a fruit-ful way"
    ),
    backstory = (
        "as a search agent, you have a good knowledge on "
        "web and for your record of doing appropiate search."
    ),
    llm = llm_2,
    tools = [search_tool],
    #verbose = True,
    allow_delegation = False
)

summary_agent = Agent(
    role = "Content Summarizer",
    goal = "summarize the given content so that the summary will give the user a good understanding about the {topic}",
    backstory = (
        "a record holder for summarizing the content and convey"
        "the results in a better and understandable way"
    ),
    llm = llm,
    #verbose = True
)


search_task = Task(
    description=(
        "you have to search the goolgle and get the content "
        "purely pertaining to the user input"
    ),
    expected_output = (
        "a content from the web that aligns with the user "
        "requirement and helpful for summary generation"
    ),
    agent = search_agent,
    #verbose = True
)

summary_task = Task(
    description = (
        "summarize the content given by the summary agent in a simple "
        "and plain 5-6 sentence summary."
    ),
    expected_output="a brief summary of the content in a plain english.",
    agent = summary_agent,
    #verbose = True
)

crew = Crew(
    agents = [search_agent,summary_agent],
    tasks = [search_task,summary_task],
    #verbose = True
)

rel = crew.kickoff(inputs = {'topic' : "Agentic AI"})

print(rel)