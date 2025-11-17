import os 
import json 
from dotenv import load_dotenv 
load_dotenv()

from crewai import LLM, Agent, Task, Crew 
from crewai_tools import EXASearchTool,ScrapeWebsiteTool

# llm = LLM(
#     model = "google/models/gemini-2.5-flash",
#     api_key = os.getenv("Google_API_Key"),
#     verbose = True
# )

# 

# llm = LLM(
#     model = 'huggingface/allenai/llama-3.1-tulu-2-8b',
#     api_key = os.getenv("HF_Token"),
#     stream = False
# )



llm = LLM(
    model = 'huggingface/meta-llama/Llama-3.1-8B-Instruct',
    api_key = os.getenv("HF_Token"),
    max_tokens = 1024,
    max_input_tokens = 4000,
    stream = False
)

# llm = LLM(
#     model = 'huggingface/allenai/llama-3.1-tulu-2-8b',
#     api_key = os.getenv("HF_Token"),
#     stream = False
# )

# llm = LLM(
#     model = 'huggingface/togethercomputer/LLaMA-2-7B-32K',
#     api_key = os.getenv("HF_Token"),
#     stream = False
# )

exa_search_tool = EXASearchTool(api_key = os.getenv("EXA_Token"))
scrape_website_tool = ScrapeWebsiteTool()


research_planner = Agent(
    role = "Research Planner",
    goal = "Analyze quiries and break them off into smaller research topics.",
    backstory = (
        "Being a Research Planner, you have a solid "
        "experience in breaking down the larger query into "
        "smaller chain topics thereby enabling clear understanding of the "
        "query and help answer better,specific and clear way."
        "\nIMPORTANT: Your output MUST be under 4000 tokens. Summarize large content into another 100 tokens."
    ),
    llm = llm,
    verbose = True
)

researcher = Agent(
    role = "Internet Researcher",
    goal = "Research in-depth all the topics.",
    backstory = (
        "you are an expert in searching the internet on the inputed topics."
        "your search potential and getting the right results, helped "
        "in getting the right responses to the topics."
        "\nIMPORTANT: Your output MUST be under 4000 tokens.Summarize large content into another 100 tokens."
    ),
    llm = llm,
    tools = [exa_search_tool,scrape_website_tool],
    verbose = True
)

fact_checker = Agent(
    role = "Fact Checker",
    goal = (
        "verify data for accuracy,identify inconsistencies "
        "and flag potential misinformation."
    ),
    backstory = (
        "as a Fact Checker agent, you master the data accuracy and flagging "
        "infodemic if any.your insights help in finding out short comings in the data "
        "and correcting the data for better results."
        "\nIMPORTANT: Your output MUST be under 4000 tokens.Summarize large content into another 100 tokens."
    ),
    llm = llm,
    tools = [exa_search_tool,scrape_website_tool],
    verbose = True
)

report_writer = Agent(
    role = "Report Writer",
    goal = "write clear,specific and well-structed reports based on gathered information.",
    backstory = (
        "you are an expert in writing who specializes in creating clear, well-structured "
        "research reports.you synthasize complex information into readable formats and "
        "always include proper citations and references."
        "\nIMPORTANT: Your output MUST be under 4000 tokens.Summarize large content into another 100 tokens."
    ),
    llm = llm,
    verbose = True
)

research_planner_task = Task(
    description = (
        "Break down the complex query into smaller component topics "
        "and create a focussed research plan.\n"
        "user query : {user_query}"
        "\nSTRICT LIMIT: Output must be less than 4000 tokens.Summarize large content into another 100 tokens."
    ),
    expected_output = (
        "a research plan with main research topics to investicate "
        "key questions to each topic, and success criteria for the research."
    ),
    agent = research_planner
)

researcher_task = Task(
    description = (
        "using the research plan,carry out intense research on web for finding the information we "
        "needed to complete this research successfully."
        "\nSTRICT LIMIT: Output must be less than 4000 tokens.Summarize large content into another 100 tokens."
    ),
    expected_output = (
        "comprehensive research data including:information for each "
        "topic ,and citations used along with source notes."
        "and, please use only 5k tokens, dont go above 5k."
    ),
    agent = researcher
)

fact_checker_task = Task(
    description = (
        "review research data, and identify any misinformation is there or not "
        "and any gaps that needed addressing"
        "\nSTRICT LIMIT: Output must be less than 4000 tokens.Summarize large content into another 100 tokens."
    ),
    expected_output = (
        "a report with all the original data you received plus "
        "any facts vs questionable information and make sure "
        "it is comprehensive and clean useful for final report generation."
        "and, please use only 5k tokens, dont go above 5k."
    ),
    agent = fact_checker
)

report_writer_task = Task(
    description = (
        "create a comprehensive report that answers the original query with clear sections,citations if any."
        "\nSTRICT LIMIT: Output must be less than 4000 tokens.Summarize large content into another 100 tokens."),
    expected_output = (
        "a final research report containing : summary, detailed findings "
        "that answer the user query,supporting evidence and analysis,"
        "complete source citations."
    ),
    agent = report_writer
)

crew = Crew(
    agents = [research_planner,researcher,fact_checker,report_writer],
    tasks = [research_planner_task,researcher_task,fact_checker_task,report_writer_task]
)

user_query = "explain the trends in AI. guide me focus areas to stay relevant."

rel = crew.kickoff(
    inputs = {
        'user_query' : user_query
    }
)