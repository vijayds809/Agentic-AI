# alert_detection_agent.py

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re

class AlertDetectionAgent:
    def __init__(self, azure_deployment, azure_endpoint, api_version, api_key):
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key
        )

    def detect_alerts(self, data_list, person_dict):
        """
        Uses Azure OpenAI LLM to perform strict exact match comparison
        of first and last names. Returns only exact matches in JSON format.
        
        Args:
            data_list (list of dict): Each dict with keys 'watchlistid', 'first_name', 'last_name'
            person_dict (dict): {'first_name': ..., 'last_name': ...}

        Returns:
            list of dict: Matching dicts from data_list with watchlistids
        """
        # Build natural language prompt for LLM to handle exact name matching
        prompt = (
            "You are a strict data matching assistant. Your task is to find all entries "
            "in the provided list whose 'first_name' AND 'last_name' exactly match those "
            "of the provided person. Only return entries that are exact matches — "
            "no partial, fuzzy, or case-insensitive matches.\n\n"
            f"Records List: {data_list}\n"
            f"Person to Match: {person_dict}\n\n"
            "Respond strictly in JSON format as a list of matching records. "
            "Return an empty list [] if no exact matches exist."
        )

        # Invoke the LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Parse and ensure JSON-like response
        try:
            match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print("Error parsing LLM output:", e)

        # Fallback when no exact JSON structure is returned
        return []
