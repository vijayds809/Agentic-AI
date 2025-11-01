import os
from dotenv import load_dotenv 
load_dotenv()
from crewai_tools import SerperDevTool

os.environ['Serper_API_Key'] = os.getenv("Serper_API_Key")

tool = SerperDevTool()