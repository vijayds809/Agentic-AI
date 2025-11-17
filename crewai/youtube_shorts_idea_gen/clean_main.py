import os
import json
from dotenv import load_dotenv
load_dotenv()

from crewai import LLM,Agent,Task,Crew 

llm = LLM(
    model = 'huggingface/meta-llama/Meta-Llama-3-8B-Instruct',
    api_key = os.getenv("HF_Token"),
    stream = False
)

creator_agent = Agent(
    role = "Youtube Shorts Ideas Creator",
    goal = (
        "plan a 1-week slate of high-retention youtube shorts about "
        "surprising origins of everyday things."
    ),
    backstory = (
        "you specialize in 30-45s micro-history that hooks fast and pays off with a twist "
        "you keeps ideas with filmable by a solo creator at home with miniman props."
    ),
    llm = llm,
)

task = Task(
    description = (
        "create a 1-week video posting plan with 5 video blueprints. "
        "Platform : Youtube Shorts (verticl 9:15, 30-45s)."
        "Niche: Micro-history of everyday things."
        "Primary goals : thumb-stop hook in first 1s and "
        "comment-biat CTA, Strong SEO-phrasing in title/caption."
        "Context : solo creator,home-filmable and no special-gear"
    ),
    expected_output =  (
        '''
Output a JSON array following the schema below, which contains a
weekly schedule and 5 video blueprints. each video blueprint should include 
{
"Videos" : [
{
"title" : "searchable,curiocity-driven title",
"hook_main" : "<=12 words, shows off fast",
"hook_alt" : "varient hook",
"visuals" : ["simple-prop","idea 2"],
"tags" : ["microhistory","everydaythings","shorts"],
"cta":"question that invites comments"
}
]
}
        '''
    ),
    agent = creator_agent
)

crew = Crew(
    agents = [creator_agent],
    tasks = [task]
)

rel = crew.kickoff()
json_output = rel.raw  #we can push this json object into another systems 
if isinstance(json_output,str):
    dict_output = json.loads(json_output)
    print(dict_output)
    print(type(dict_output))
else:
    print(json_output)
    print(type(rel.raw))