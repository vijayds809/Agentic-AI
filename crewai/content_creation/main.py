import os
import yaml
import json
import textwrap
from IPython.display import display,Markdown
from dotenv import load_dotenv 
load_dotenv()
from crewai import LLM,Agent,Task,Crew 
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool
from pydantic import BaseModel,Field 
from typing import List,Tuple,Set,Dict,Optional

search_tool = SerperDevTool()
web_content_tool = ScrapeWebsiteTool()
rag_over_web_tool = WebsiteSearchTool()

google_llm = LLM(
    model = 'google/models/gemini-2.5-flash',
    api_key = os.getenv("Google_API_Key"),
    stream = False
)

meta_llm = LLM(
    model = 'huggingface/meta-llama/Meta-Llama-3-8B-Instruct',
    api_key = os.getenv("HF_Token"),
    stream = False
)

from pydantic import BaseModel, Field
from typing import List

class SocialMediaPost(BaseModel):
    platform: str = Field(..., description="The social media platform where the post will be published (e.g., Twitter, LinkedIn).")
    content: str = Field(..., description="The content of the social media post, including any hashtags or mentions.")

class ContentOutput(BaseModel):
    article: str = Field(..., description="The article, formatted in markdown.")
    social_media_posts: List[SocialMediaPost] = Field(..., description="A list of social media posts related to the article.")

files = {
    'agents': './agents.yaml',
    'tasks': './tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']


# Creating Agents
market_news_monitor_agent = Agent(
    config=agents_config['market_news_monitor_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=meta_llm,
)

data_analyst_agent = Agent(
    config=agents_config['data_analyst_agent'],
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm=meta_llm,
)

content_creator_agent = Agent(
    config=agents_config['content_creator_agent'],
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm = meta_llm
)

quality_assurance_agent = Agent(
    config=agents_config['quality_assurance_agent'],
    llm = google_llm
)

# Creating Tasks
monitor_financial_news_task = Task(
    config=tasks_config['monitor_financial_news'],
    agent=market_news_monitor_agent
)

analyze_market_data_task = Task(
    config=tasks_config['analyze_market_data'],
    agent=data_analyst_agent
)

create_content_task = Task(
    config=tasks_config['create_content'],
    agent=content_creator_agent,
    context=[monitor_financial_news_task, analyze_market_data_task]
)

quality_assurance_task = Task(
    config=tasks_config['quality_assurance'],
    agent=quality_assurance_agent,
    output_pydantic=ContentOutput
)

# Creating Crew
content_creation_crew = Crew(
    agents=[
        market_news_monitor_agent,
        data_analyst_agent,
        content_creator_agent,
        quality_assurance_agent
    ],
    tasks=[
        monitor_financial_news_task,
        analyze_market_data_task,
        create_content_task,
        quality_assurance_task
    ],
    verbose=True
)
print("Crew Kickoff started in remote")
result = content_creation_crew.kickoff(inputs={
  'subject': 'Inflation in the US and the impact on the stock market in 2024'
})

output = result.model_dump_json()
with open("output.json","w",encoding = 'utf-8') as f:
    json.dump(output,f,indent = 4,ensure_ascii = False)
# posts = result.pydantic['social_media_posts']
# for post in posts:
#     platform = post['platform']
#     content = post['content']
#     print(platform)
#     print("\n")
#     wrapped_content = textwrap.fill(content,width = 50)
#     print(wrapped_content)
#     print('-'*50)

# display(Markdown(result.pydantic['article']))
#print("ALL Done!")
