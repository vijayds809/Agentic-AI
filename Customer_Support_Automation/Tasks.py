from crewai import Task 
from Agents import support_agent, support_quality_assurance_agent
from Tools import docs_scrape_tool

inquiry_resolution = Task(
    description = (
        "{customer} just reached out with super important ask.\n"
        "{inquiry}\n\n"
        "{person} from {customer} is the who reached out. "
        "make sure to use everything you know "
        "to provide the best possible support. "
        "you must strive to provide a complete "
        "and accurate response to the customer inquiry."
    ),
    expected_output = (
        "a detailed informative response to the customer inquiry "
        "that addresses all aspects of the customer's question \n"
        "the response should include references to everything you used "
        "for finding the answer including external data or solutions. "
        "ensure the answer is complete leaving no questions unanswered. "
        "the tone should be friendly, professional, and empathetic."
    ),
    tools = [docs_scrape_tool],
    agent = support_agent
)

quality_assurance_review = Task(
    description = (
        "Review the response provided by the senior support representative "
        "ensure that the answer is comprehensive, accurate, and adheres to "
        "high standards of customer support."
        "verify that all aspects of the customer's inquiry have been addressed "
        "throughly with a helpful and friendly tone."
        "check for references to any external data or solutions used to"
        "find the information."
        "ensure the response is well-supported and leaves no questions unanswered."
    ),
    expected_output = (
        "a final, detailed and informative response "
        "ready to be sent to the customer.\n"
        "this response should fully address the customer's inquiry "
        "incorporating all relevant feedback and improvements.\n"
        "dont be too formal, we all are cool and chill company "
        "but maintain professional and friendly tone throughout the response."
    ),
    agent = support_quality_assurance_agent
)