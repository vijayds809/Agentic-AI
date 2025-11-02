from Agents import support_agent, support_quality_assurance_agent
from Tools import docs_scrape_tool
from Tasks import inquiry_resolution, quality_assurance_review
from crewai import Crew 

crew = Crew(
    agents = [support_agent, support_quality_assurance_agent],
    tasks = [inquiry_resolution, quality_assurance_review],
    verbose =True,
    memory = True
)

inputs = {
    "person" : "Vijay Kumar B N",
    "customer" : "DeepLearningAI",
    "inquiry" : "i need  help in setting up of crew"
                "and kicking it off, specifically"
                "how can i add memory to my crew?"
                "can you provide step by step instructions?"
}

res = crew.kickoff(inputs = inputs)