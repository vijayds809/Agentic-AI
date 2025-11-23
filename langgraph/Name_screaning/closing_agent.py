# closing_agent.py

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re


class ClosingAgent:
    def __init__(self, azure_deployment, azure_endpoint, api_version, api_key):
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key
        )

    def close_alerts(self, list_of_dicts, customer_details):
        """
        Uses Azure OpenAI LLM to classify alerts into priorities and generate
        disposition summaries based on similarity scores and customer details.

        Args:
            list_of_dicts (list): list with similarity_score and watchlistid
            customer_details (dict): customer info for identifying missing fields

        Returns:
            list[dict]: JSON-based structured output for each alert
        """

        # Prettify the prompt
        prompt = (
            "You are a compliance disposition assistant. You will receive a list of alerts, "
            "each having a similarity score (1-100) between the customer's information and "
            "the watchlist entry. Based on this score, perform the following tasks:\n\n"

            "Rules:\n"
            "- Priority = 'Low' if similarity_score < 30\n"
            "- Priority = 'Medium' if 30 <= similarity_score <= 90\n"
            "- Priority = 'High' if similarity_score > 90\n"
            "- Disposition Status:\n"
            "    * 'Closed' for Low priority\n"
            "    * 'Further Review' for Medium priority\n"
            "    * 'Escalate' for High priority\n\n"

            f"Customer Details: {customer_details}\n\n"

            "For Medium priority, check for missing or empty fields in the provided "
            "customer details and mention them in the disposition summary.\n\n"

            "Generate output as a JSON list. Each element must include:\n"
            "{\n"
            "    'watchlistid': ..., \n"
            "    'Similarity': 'XX%', \n"
            "    'priority': 'Low/Medium/High', \n"
            "    'dispositionstatus': 'Closed/Further Review/Escalate', \n"
            "    'dispositionsummary': '...'\n"
            "}\n\n"

            "Example:\n"
            "[{\n"
            "  'watchlistid': 'WL123',\n"
            "  'Similarity': '85%',\n"
            "  'priority': 'Medium',\n"
            "  'dispositionstatus': 'Further Review',\n"
            "  'dispositionsummary': 'The alert has been escalated for further review due to missing PAN and phone number.'\n"
            "}]\n\n"

            "Analyze the following alerts and produce JSON output accordingly:\n"
            f"{list_of_dicts}\n"
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Extract JSON response from LLM
        try:
            match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if match:
                structured_output = json.loads(match.group())
                return structured_output
        except Exception as e:
            print("Error parsing LLM structured output:", e)

        return []


# Example usage
# if __name__ == "__main__":
#     agent = ClosingAgent(
#         azure_deployment="your-deployment",
#         azure_endpoint="https://your-resource.openai.azure.com/",
#         api_version="2024-05-01",
#         api_key="your-azure-key"
#     )

#     alerts = [
#         {"watchlistid": "WL123", "similarity_score": 25},
#         {"watchlistid": "WL125", "similarity_score": 65},
#         {"watchlistid": "WL127", "similarity_score": 95}
#     ]

#     customer_details = {
#         "first_name": "vijay",
#         "last_name": "kumar",
#         "country": "India",
#         "pan": ""
#     }

#     output = agent.close_alerts(alerts, customer_details)
#     print(json.dumps(output, indent=2))
