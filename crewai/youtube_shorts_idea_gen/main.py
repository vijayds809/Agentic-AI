import os
from dotenv import load_dotenv
load_dotenv()

from crewai import LLM,Agent,Task,Crew 

llm = LLM(
    model = 'huggingface/meta-llama/Meta-Llama-3-8B-Instruct',
    api_key = os.getenv("HF_Token"),
    stream = False
)

content_creator_assistant = Agent(
    role = "Content Creator Assistant",
    goal = "come up with ideas for creating youtube shorts.",
    backstory = (
        "you are an expert in coming up with ideas "
        "for generating video content."
    ),
    llm = llm
)

task = Task(
    description = "come up with 5 new youtube shorts video ideas.",
    expected_output = "list of ideas.",
    agent = content_creator_assistant
)

crew = Crew(
    agents = [content_creator_assistant],
    tasks = [task]
)

rel = crew.kickoff()
print(rel.raw)
print(type(rel.raw))