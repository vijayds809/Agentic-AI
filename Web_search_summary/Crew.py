from crewai import Crew,Process 
from Agents import researcher_agent,writer_agent
from Tasks import research_task,writing_task

crew = Crew(
    agents = [researcher_agent, writer_agent],
    tasks = [research_task, writing_task],
    process = Process.sequential
)

rel = crew.kickoff(inputs ={
    "topic":"AI in disaster management"
})

print(rel)