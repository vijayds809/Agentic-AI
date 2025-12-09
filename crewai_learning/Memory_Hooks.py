import os 
import re
import yaml 
from IPython.display import Markdown
from crewai import LLM,Agent,Task,Crew 
from crewai_tools import EXASearchTool,ScrapeWebsiteTool
from chromadb.utils import embedding_functions 
from dotenv import load_dotenv 
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

llm = LLM(
    model = 'huggingface/meta-llama/Llama-3.1-8B-Instruct',
    api_key = os.getenv("HF_Token"),
    max_tokens = 1024,
    max_input_tokens = 4000,
    stream = False
)



# llm = LLM(
#     # base_url = os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key = os.getenv("AZURE_OPENAI_API_KEY"),
#     # api_version= os.getenv("AZURE_OPENAI_API_VERSION"),
#     #deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
#     model = os.getenv("Model"),
# )

local_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
exa_search_tool = EXASearchTool(api_key = os.getenv("EXA_Token"))
website_scrape_tool = ScrapeWebsiteTool()

with open("configs/agents.yaml","r",encoding = 'utf-8') as file:
    agents_config = yaml.safe_load(file)

research_planner_agent = Agent(
    config = agents_config["research_planner"],
    llm = llm,
    verbose = True
)

internet_researcher_agent = Agent(
    config = agents_config["internet_researcher"],
    llm = llm,
    tools = [exa_search_tool,website_scrape_tool],
    verbose = True
)

fact_checker_agent = Agent(
    config = agents_config["fact_checker"],
    llm = llm,
    tools = [exa_search_tool,website_scrape_tool],
    verbose = True
)

report_writer_agent = Agent(
    config = agents_config['report_writer'],
    llm = llm,
    verbose = True
)

#adding code_guardrail to check the final output has summary section,references etc 

def write_report_guardrail(output):
    try:
        output = output if type(output) == str else output.raw 
    except Exception as e:
        return (False,(
            "Error retrieving the raw output"
            f"\n{str(e)}\n"
        ))
    output_lower = output.lower()

    if not re.search(r'#+.*summary',output_lower):
        return (False,
                "Report must include summary section")
    
    if not re.search(r'#+.*insights|#+.*references',output_lower):
        return (False,
                "Report must include insights/references section")
    
    if not re.search(r'#+.*citations|#+.*references',output_lower):
        return (False,
                "Report must include citations/references section")
    
    return (True,output)


with open("configs/tasks.yaml","r",encoding = 'utf-8') as file:
    tasks_config = yaml.safe_load(file)

create_search_plan_task = Task(
    config = tasks_config["create_research_plan"],
    agent = research_planner_agent
)

gather_research_data_task = Task(
    config = tasks_config["gather_research_data"],
    agent = internet_researcher_agent
)

verify_information_quality_task = Task(
    config = tasks_config["verify_information_quality"],
    agent = fact_checker_agent
)

write_final_report_task = Task(
    config = tasks_config["write_final_report"],    
    agent = report_writer_agent,
    guardrails = [write_report_guardrail]
)


def save_file_hook(result):
    try:
        if hasattr(result,'tasks_output') and result.tasks_output:
            report_content = result.tasks_output[-1].raw
        else:
            report_content = str(result)
        filename = f"research_report.md"
        with open(filename,"w",encoding = "utf-8") as file:
            file.write(report_content)
        print("report successfully saved to ",filename)
    except Exception as e:
        print("Error saving the report: ",str(e))


crew = Crew(
    agents = [research_planner_agent,
              internet_researcher_agent,    
                fact_checker_agent,
                report_writer_agent
],

    tasks = [create_search_plan_task,
             gather_research_data_task, 
             verify_information_quality_task,
             write_final_report_task],
    memory = True,
    after_kickoff_callbacks = [save_file_hook],
    memory_config={
        "embedder": local_embedder
    }
)

inputs = {
    "user_query" : "Evaluate the top five AI emerging tools for automating competative market analysis."
}

rel = crew.kickoff(inputs = inputs)
