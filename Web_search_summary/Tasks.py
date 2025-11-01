from crewai import Task 
from Tools import tool
from Agents import researcher_agent as researcher
from Agents import writer_agent as writer

research_task = Task(
    description = (
        "Itentify the next big thing in the {topic}"
        "focus on identifying pros and cons and the ovarall narrative"
        "your final report should clearly articulate the key points"
        "its market oppertunities and potential challenges."
    ),
    expected_output = "a comprehensive 3 paragraph long report on the {topic}",
    agent = researcher,
    tools = [tool]
)

writing_task = Task(
    description = (
        "compose an insightful article on the {topic}"
        "focus on latest trends and how its impacting various industries"
        "this article should be easy to understand, engaging ang positive"
    ),
    expected_output = "a 4 paragraph long article on the {topic} as a markdown file",
    tools = [tool],
    agent = writer,
    async_execution=False,
    output_file = "Final_Report.md"
)