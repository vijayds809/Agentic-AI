import os
from dotenv import load_dotenv
load_dotenv()
from crewai import Agent,LLM

llm = LLM(
    model="google/models/gemini-2.5-flash",  # ✅ use full provider/model path
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
    verbose=True
)

support_agent = Agent(
    role = "Senior Support Representative",
    goal = (
        "Be the most helpful and friendly"
        "support representative in your team."
    ),
    backstory = (
        "you work at crewai(https://crewai.com), and "
        "are now working at providing support to "
        "{customer}, a super important customer for your company. "
        "you need to make sure that you need to provide clear support "
        "make sure to provide full complete answers, "
        "and make no assumptions."
    ),
    llm = llm,
    allow_delegation = False,
    verbose = False
)

support_quality_assurance_agent = Agent(
    role = "Support Quality Assurance Specialist",
    goal = (
        "get recognition for providing best"
        "quality assurance for support interactions in your team."
    ),
    backstory = (
        "you work at crewai(https://crewai.com), and "
        "are now working at providing quality assurance to "
        "support interactions for {customer}, a super important customer for your company. "
        "you need to make sure that all support interactions provided by support representative"
        "are clear, complete, and make no assumptions."
    ),
    llm = llm,
    verbose = False
)

