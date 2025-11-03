import os 
from dotenv import load_dotenv
load_dotenv()
from crewai import LLM,Agent,Task,Crew 
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool, DirectoryReadTool, FileReadTool


os.environ['Serper_API_Key'] = os.getenv("Serper_API_Key")

#built-in tools
google_search_tool = SerperDevTool()                          
directory_read_tool = DirectoryReadTool(directory="./instructions")  # these 3 are in-build crewai tools
file_read_tool = FileReadTool()


#creating custom tool in crewai
class SentimentAnalysisTool(BaseTool):
    name:str = "Sentiment Analysis Tool"
    description:str= (
        "analyze the sentiment of the text "
        "to ensure positive and engaging communication."
    )
    def _run(self,text:str)->str:
        return "Positive"
    
sentiment_analysis_tool = SentimentAnalysisTool()


llm = LLM(
    model="google/models/gemini-2.5-flash",  # ✅ use full provider/model path
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
    verbose=True
)

sales_rep_agent = Agent(
    role = "Sales Representative",
    goal = (
        "Identify high-value leads that match"
        " our ideal customer profile"
    ),
    backstory = (
        "as a part of dynamic sales team at the crewai"
        "your mission is to scour"
        "the digital landscape for potential leads"
        " armed with cutting-edge tools"
        "and strategic mindset, you analyze data, "
        "trends, and interactions to"
        "unearth oppertunities that others might overlook "
        "your work is crucial for paving the way "
        "for meaningful engagements and driving company's growth."
    ),
    llm = llm,
    allow_delegation = False,
    verbose = True
)


lead_sales_rep_agent = Agent(
    role = "Lead Sales Representative",
    goal = "Nurture leads with personalized and compelling outreach",
    backstory = (
        "with the vibrant ecosystem of crewai's sales department, "
        "you stand out as a bridge between potential clients "
        "and the solutions they needed. "
        "by creating engaging and personal narratives "
        "you not only inform leads about our offerings "
        "but also make them feel seen and heard. "
        "your role is important in transforming interest into commitment and action, "
        "guiding leads through the journey from curiosity to commitment."
    ),
    llm = llm,
    allow_delegation = False,
    verbose = True
)


lead_profiling_task = Task(
    description = (
        "conduct an indepth analysis of {lead_name} "
        "a company in the {industry} sector "
        "that recently showed interest in our solutions. "
        "utilize all available data sources "
        "to compile a detailed profile "
        "focusing on key decision makers, recent business "
        "developments, and potential needs "
        "that aligh with our offerings. "
        "this task is crucial for tailoring "
        "our engaging stratagy effectively.\n"
        "dont make assumptions and"
        "you only use information you absolutely sure about."
    ),
    expected_output = (
        "a comprehensive report on {lead_name} "
        "including company background "
        "key personnal, recent milestones and identified needs"
        "highlight potential areas where "
        "our solutions provide value "
        "and suggest personalized engagement stratagies"
    ),
    agent =  sales_rep_agent,
    tools = [directory_read_tool,file_read_tool,google_search_tool]
)

personalized_outreach_task = Task(
    description = (
        "using the insights gathering from the "
        "lead profiling report on {lead_name} "
        "craft a personalized outreach campaigh "
        "aimed at {key_decision_maker}"
        "the {position} of {lead_name}."
        "the campaigh should address their recent {milestone} "
        "and how our solutions can support their goals. "
        "your communication must resonate "
        "with {lead_name}'s company culture,values and "
        "demonstrating a deep understanding of "
        "their business and leads.\n"
        "dont make assumptions and only "
        "use information you absolutely sure about."

    ),
    expected_output = (
        "a series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically tragetting {key_decision_maker} "
        "each draft should include "
        "a compelling narrative that connects our solutions "
        "with their recent achievements and future goals "
        "ensure the tone is engaging,friendly and professional "
        "and aligned with {lead_name}'s corporate identity."
    ),
    agent = lead_sales_rep_agent,
    tools = [sentiment_analysis_tool,google_search_tool],
    verbose = True
)

crew = Crew(
    agents = [sales_rep_agent,lead_sales_rep_agent],
    tasks = [lead_profiling_task,personalized_outreach_task],
    memory = True,
    verbose = True
)

inputs = {
    "lead_name" : "Deep Learning AI",
    "industry" : "Online Learning Platform",
    "key_decision_maker" : "Andrew NG",
    "position" : "CEO",
    "milestone" : "product launch"
}

rel = crew.kickoff(inputs = inputs)