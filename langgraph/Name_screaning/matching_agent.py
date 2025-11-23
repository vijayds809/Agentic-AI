# matching_agent.py

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re


class MatchingAgent:
    def __init__(self, azure_deployment, azure_endpoint, api_version, api_key):
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key
        )

    def match(self, input_dict, list_of_dicts):
        """
        Uses Azure OpenAI LLM to compute similarity score (1-100) between input_dict
        and each dict in list_of_dicts. Returns top 5 matched records with
        'similarity_score' added to each dict.
        """

        prompt = (
            "You are a smart data comparison assistant. Compare a given input record "
            "with a list of records and assign a similarity score for each entry "
            "from 1 to 100. The score reflects how closely the values of corresponding "
            "keys (like name, country, ID, etc.) match.\n\n"
            f"Input Record: {input_dict}\n"
            f"List of Records: {list_of_dicts}\n\n"
            "Return a JSON array where each record includes all original keys plus an "
            "extra key 'similarity_score' representing the integer score (1-100). "
            "Use higher scores for records that are more similar.\n\n"
            "Example Expected JSON Format:\n"
            "[\n"
            "  {'watchlistid': 'WL123', 'first_name': 'vijay', 'last_name': 'kumar', 'similarity_score': 95},\n"
            "  {'watchlistid': 'WL124', 'first_name': 'ajit', 'last_name': 'verma', 'similarity_score': 45}\n"
            "]"
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if match:
                enhanced_records = json.loads(match.group())

                # Sort records by similarity_score descending
                sorted_list = sorted(enhanced_records, key=lambda x: x.get('similarity_score', 0), reverse=True)
                top_5 = sorted_list[:5]
                top_confidence_score = top_5[0].get('similarity_score', 0) if top_5 else 0
                top_most_entry = top_5[0] if top_5 else {}

                return top_5, top_confidence_score, top_most_entry
        except Exception as e:
            print("Error parsing LLM response:", e)

        # fallback return if LLM output invalid
        return [], 0, {}


# Example Usage
# if __name__ == "__main__":
#     agent = MatchingAgent(
#         azure_deployment="your-deployment",
#         azure_endpoint="https://your-resource.openai.azure.com/",
#         api_version="2024-05-01",
#         api_key="your-azure-api-key"
#     )

#     input_dict = {"first_name": "vijay", "last_name": "kumar", "country": "India", "pan": "IGSPK1234K"}
#     list_of_dicts = [
#         {"watchlistid": "WL123", "first_name": "vijay", "last_name": "kumar", "country": "India", "pan": "IGSPK1234K"},
#         {"watchlistid": "WL124", "first_name": "ajay", "last_name": "singh", "country": "Bangladesh", "pan": "BGLPL124L"},
#         {"watchlistid": "WL125", "first_name": "vijay", "last_name": "kumara", "country": "India", "pan": "IGSPK9876Z"},
#     ]

#     top_5, top_score, top_entry = agent.match(input_dict, list_of_dicts)
#     print("Top 5 Matches:", top_5)
#     print("Top Confidence Score:", top_score)
#     print("Top Entry:", top_entry)
