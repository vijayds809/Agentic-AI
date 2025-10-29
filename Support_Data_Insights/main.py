import os 
import yaml
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display,Markdown,Image
from crewai import LLM,Agent,Task,Crew 
from crewai_tools import FileReadTool
from dotenv import load_dotenv 
load_dotenv()

csv_tool = FileReadTool(file_path = "./customer_issues.csv")

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

files = {
    "agents" : "configs/Agents.yaml",
    "tasks" : "configs/Tasks.yaml",
}

config = {}

for config_type,file_path in files.items():
    with open(file_path) as file:
        config[config_type] = yaml.safe_load(file)

agents_config = config['agents']
tasks_config = config['tasks']

suggestion_generation_agent = Agent(
    config = agents_config['suggestion_generation_agent'],
    tools = [csv_tool]
)

reporting_agent = Agent(
    config = agents_config['reporting_agent'],
    tools = [csv_tool]
)

chart_generation_agent = Agent(
    config = agents_config['chart_generation_agent'],
    allow_code_execution = True                            #we need docker installations to execute this agent as it invloves code genration and execution in a contained environment(container)
)

suggestion_generation = Task(
    config = tasks_config['suggestion_generation'],
    agent = suggestion_generation_agent
)

table_generation = Task(
    config = tasks_config['table_generation'],
    agent = reporting_agent
)

chart_generation = Task(
    config = tasks_config['chart_generation'],
    agent = chart_generation_agent
)

final_report_assembly = Task(
    config = tasks_config['final_report_assembly'],
    agent = reporting_agent,
    context = [suggestion_generation, table_generation, chart_generation]
)

support_report_crew = Crew(
    agents = [suggestion_generation_agent, reporting_agent, chart_generation_agent],
    tasks = [suggestion_generation, table_generation, chart_generation, final_report_assembly],
    verbose = True
)

support_report_crew.test(n_iterations=1,eval_llm=llm)  #for testing the crew efficiency
support_report_crew.train(n_iterations=1,filename="training.pkl")

rel = support_report_crew.kickoff()

display(Markdown(rel.raw))