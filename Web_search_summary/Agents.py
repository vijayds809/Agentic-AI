import os
from dotenv import load_dotenv 
load_dotenv()

from crewai import Agent,LLM 
from langchain_google_genai import ChatGoogleGenerativeAI 
from Tools import tool

#os.environ["LITELLM_PROVIDER"] = "google"

llm = LLM(
    model="google/models/gemini-2.5-flash",  # ✅ use full provider/model path
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
    verbose=True
)

researcher_agent = Agent(
    role = "Senior Researcher",
    goal = "uncover ground breaking technologies in {topic}",
    backstory = (
        "Driven by curiosity, you are at the forefront of"
        "innovation,eager to explore and share the knowledge that could change"
        "the world."
    ),
    llm = llm,
    tools = [tool],
    allow_delegation = True,
    verbose = True,
    memory = True
)

writer_agent = Agent(
    role = "Content Writer",
    goal = "narrate compelling tech stories about the {topic}",
    backstory = (
        "with a flair for simplifying the complex topics, your craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    llm = llm,
    tools = [tool],
    allow_delegation = False,
    memory = True,
    verbose = True
)

