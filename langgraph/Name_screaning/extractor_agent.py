# extractor_agent_llm_azure.py

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re

class ExtractorAgentLLMAzure:
    def __init__(self, azure_deployment, azure_endpoint, api_version, api_key):
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key
        )

    def extract(self, input_dict):
        """
        Extract first and last name fields from input_dict using Azure OpenAI.
        Returns a simple dictionary: {'first_name':..., 'last_name':...}
        """
        prompt = (
            "Extract 'first_name' and 'last_name' from the following dictionary. "
            "Respond ONLY with a Python dictionary in this format: "
            "{'first_name': ..., 'last_name': ...}\n"
            f"Input: {input_dict}"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        if match:
            try:
                # Convert single quotes to double quotes for JSON parsing
                extracted = json.loads(match.group().replace("'", '"'))
                return extracted
            except Exception:
                pass
        return {"first_name": "", "last_name": ""}
