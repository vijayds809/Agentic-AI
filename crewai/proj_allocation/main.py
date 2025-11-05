import os
import json 
import yaml
from crewai import LLM,Agent,Crew,Task 
from crewai.tools import BaseTool 
from crewai_tools import SerperDevTool,ScrapeWebsiteTool,FileReadTool,DirectoryReadTool
from typing import List 
from pydantic import BaseModel,Field
from dotenv import load_dotenv 
load_dotenv()

llm = LLM(
    model = "google/models/gemini-2.5-flash",
    api_key = os.getenv("Google_API_Key"),
    verbose = True
)

files = {
    'agents' : './Agents.yaml',
    'tasks' : './Tasks.yaml'
}

configs = {}
for config_type,file_path in files.items():
    with open(file_path) as file:
        configs[config_type] = yaml.safe_load(file)

agents_config = configs['agents']
tasks_config = configs['tasks']

class TaskEstimate(BaseModel):
    task_name:str = Field(...,description = "Name of the task")
    estimated_time_hours:float = Field(...,description = "estimated time to do or completed the task in hours")
    required_resources:List[str] = Field(...,description = "List of resources required for doing the tasks in the project")

class Milestone(BaseModel):
    milestone_name:str = Field(...,description = "Name of the Milestone")
    tasks:List[str] = Field(...,description = "list of taskids associated with the tasks")

class ProjectPlan(BaseModel):
    tasks:List[TaskEstimate] = Field(...,description = "list of tasks")
    milestones:List[Milestone] = Field(...,description = 'list of project milestones')


project_planning_agent = Agent(
    config = agents_config['project_planning_agent'],
    llm = llm 
)

estimation_agent = Agent(
    config = agents_config['estimation_agent'],
    llm = llm
)

resource_allocation_agent = Agent(
    config = agents_config['resource_allocation_agent'],
    llm = llm
)

task_breakdown = Task(
    config = tasks_config['task_breakdown'],
    agent = project_planning_agent
)

time_resource_estimation = Task(
    config = tasks_config['time_resource_estimation'],
    agent = estimation_agent
)

resource_allocation = Task(
    config = tasks_config['resource_allocation'],
    agent = resource_allocation_agent,
    output_pydantic = ProjectPlan
)

crew = Crew(
    agents = [project_planning_agent,estimation_agent,resource_allocation_agent],
    tasks = [
        task_breakdown,
        time_resource_estimation,
        resource_allocation
    ],
    verbose = True
)


project = "Website"
industry = "Technology"
project_objectives = "create a website for a small business"
team_members = """
-John mick (Project Managet)
-John Doe (Software Engineer)
-Bob Smith (Designer)
-Tom Brown (QA Engineer)
"""
project_requirements = """
-create a responsible design that works for desktop and mobile sets
-implement visually appealing and UI with a clean look and design
-develop user friendly navigation
-ensure fast loading times and optimize for search engines
-integrate social media links and sharing capabilities 
"""

input_data = {
    'project_type' : project,
    'project_objectives':project_objectives,
    'industry':industry,
    'team_members':team_members,
    'project_requirements':project_requirements
}

rel = crew.kickoff(inputs = input_data)

token_data = {
    "input_tokens" : crew.usage_metrics.prompt_tokens,
    "output_tokens" : crew.usage_metrics.completion_tokens
}

print("**************Token Usage Data**********")
print("\n")
print(token_data)