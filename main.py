import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os 
from dotenv import load_dotenv
load_dotenv()   

from crewai import Agent, Task, Crew, LLM 

# These must match your Azure OpenAI setup exactly:
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

agent = Agent(
    role="assistant",
    goal="answer user questions as an assistant",
    backstory="You are a helpful AI assistant that answers user questions clearly and concisely.",
    llm=llm
)

task = Task(
    description="What is Agentic AI?",
    expected_output="A brief, plain-language explanation of what Agentic AI is and how it works.",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task]
)

response = crew.kickoff()
print("Response:", response)
