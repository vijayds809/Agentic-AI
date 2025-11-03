import os 
import warnings 
warnings.filterwarnings('ignore')
from dotenv import load_dotenv 
load_dotenv()
from crewai import LLM, Agent, Task, Crew 
from crewai_tools import SerperDevTool,ScrapeWebsiteTool
from pydantic import BaseModel

class VenueDetails(BaseModel):
    name : str
    address : str 
    capacity : int
    booking_status : str

llm = LLM(
    model = "google/models/gemini-2.5-flash", 
    api_key = os.getenv("Google_API_Key"),
    temperature = 0.7,
    verbose = True
)

serper_api_key = os.getenv("Serper_API_Key")

google_search_tool = SerperDevTool()
web_scrape_tool = ScrapeWebsiteTool()

venue_coordinator = Agent(
    role = "Venue Coordinator",
    goal = (
        "Identify and book the venue"
        "based on the event requirement"
    ),
    backstory = (
        "with a keen sense of space "
        "and understanding of event logistics "
        "you excel at finding and securing "
        "the perfect venue the fits the event's theme "
        "size, and budget constraints."
    ),
    llm = llm,
    tools = [google_search_tool,web_scrape_tool],
    verbose = True
)

logistics_manager = Agent(
    role = "Logistics Manager",
    goal = (
        "manage all logistics for the event "
        "including catering and equipment"
    ),
    backstory = (
        "organized and detail-oriented "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup"
        "is flawlessly executed to create a seamless experience."
    ),
    llm = llm,
    tools = [google_search_tool,web_scrape_tool],
    verbose = True
)

marketing_communication_agent = Agent(
    role = "Marketing and Communication Agent",
    goal = "Effectively market the event and communicate with participents.",
    backstory = (
        "create and communicate, "
        "you craft compelling messages and "
        "engage with potential attendes "
        "to maximize event exposure and participation."
    ),
    llm = llm,
    tools = [google_search_tool,web_scrape_tool],
    verbose = True
)

venue_task = Task(
    description = (
        "find the venue in {event_city} "
        "that meets criteria for {event_topic}"
    ),
    expected_output = (
        "all the details of a specifically choosen venue "
        "you found to accommodate the event."
    ),
    agent = venue_coordinator,
    human_input = True,
    output_json = VenueDetails,
    output_file = "venue_details.json",
)

logistics_task = Task(
    description = (
        "coordinate catering and equipment for the event "
        "with {expected_participents} participents "
        "on {date}"
    ),
    expected_output = (
        "confirmation of all logistics arrangements "
        "including catering and equipment set up."
    ),
    human_input = True,
    async_execution = False,
    agent = logistics_manager
)

marketing_task = Task(
    description = (
        "promote the {event_topic} "
        "aiming to engage at least "
        "{expected_participents} potential attendees."
    ),
    expected_output = (
        "report on marketing activities "
        "and attendee engagement formatted as markdown."
    ),
    async_execution = True,
    agent = marketing_communication_agent,
    output_file = "marketing_strategy.md"
)

crew = Crew(
    agents = [venue_coordinator,logistics_manager,marketing_communication_agent],
    tasks = [venue_task,logistics_task,marketing_task],
    verbose = True
)


event_details = {
    "event_topic" : "Tech Innovation Conference",
    "event_description" : "a gathering of tech innovators and industry leaders to explore future technologies",
    "event_city" : "San Francisco",
    "date" : "2025-04-22",
    "expected_participents" : 500,
    "budget" : 20000,
    "venue_type" : "conference hall"
}

rel = crew.kickoff(inputs= event_details)
