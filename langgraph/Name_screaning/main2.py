# main_runner_langgraph.py

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List

from extractor_agent import ExtractorAgentLLMAzure
from alert_detection_agent import AlertDetectionAgent
from matching_agent import MatchingAgent
from closing_agent import ClosingAgent

import os
from dotenv import load_dotenv
load_dotenv()

State = Dict[str, Any]

def extractor_node(state: State) -> State:
    extractor = ExtractorAgentLLMAzure(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    extracted = extractor.extract(state.get("input_dict", {}))
    state["extracted"] = extracted
    return state

def alert_detection_node(state: State) -> State:
    alert_detector = AlertDetectionAgent(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    # Call LLM-based exact match detection
    exact_matches = alert_detector.detect_alerts(
        state.get("list_dicts_for_alert", []),
        state.get("extracted", {})
    )
    # Save exact matched records under 'exact_matches' key
    state["exact_matches"] = exact_matches
    return state

def matching_node(state: State) -> State:

    matcher = MatchingAgent(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Get exact matched minimal records from alert detection node
    matched_records_minimal = state.get("exact_matches", [])

    # Build set of matched watchlist ids
    matched_watchlist_ids = {rec['watchlistid'] for rec in matched_records_minimal}

    # Filter full records that match those watchlist ids
    full_records = state.get("full_records", [])
    matched_full_records = [rec for rec in full_records if rec.get('watchlistid') in matched_watchlist_ids]

    top_5, top_score, top_entry = matcher.match(
        state.get("person_dict_for_matching", {}),
        matched_full_records
    )
    state["top_5_matched"] = top_5
    state["top_confidence_score"] = top_score
    state["top_most_entry"] = top_entry
    return state



def closing_node(state: State) -> State:
    closer = ClosingAgent(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    output = closer.close_alerts(
        state.get("top_5_matched", []),
        state.get("customer_details_for_closing", {})
    )
    state["closing_output"] = output
    return state

def main():
    graph_builder = StateGraph(State)

    graph_builder.add_node("extractor_agent", extractor_node)
    graph_builder.add_node("alert_detection_agent", alert_detection_node)
    graph_builder.add_node("matching_agent", matching_node)
    graph_builder.add_node("closing_agent", closing_node)

    graph_builder.add_edge(START, "extractor_agent")
    graph_builder.add_edge("extractor_agent", "alert_detection_agent")
    graph_builder.add_edge("alert_detection_agent", "matching_agent")
    graph_builder.add_edge("matching_agent", "closing_agent")
    graph_builder.add_edge("closing_agent", END)

    graph = graph_builder.compile()

    # Initial empty inputs - fill these with actual data later
    initial_state = {
        "input_dict": {
            'lastname': 'Al-Fayez',
            'firstname': 'Waleed',
            'middlename': 'Abdullah',
            'aliasname': 'Waleed Abdullah Al-Fayez',
            'dateofbirth': 'NaT',
            'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
            'residence': 'SA',
            'citizenship': 'SA',
            'address_street': '23 Al-Masjid St, Riyadh',
            'entity': 'Entity'
        },  #input

        "list_dicts_for_alert": [
            {
                "watchlistid" : "WL123",
                "first_name" : "Waleed",
                "last_name" : "Al-Fayez"
            },
            {
                "watchlistid" : "WL124",
                "first_name" : "ajay",
                "last_name" : "singh"
            },
            {
                "watchlistid" : "WL125",
                "first_name" : "Waleed",
                "last_name" : "Al-Fayez"
            },
            {
                "watchlistid" : "WL126",
                "first_name" : "vinay",
                "last_name" : "kumar"
            },
            {
                "watchlistid" : "WL127",
                "first_name" : "vikky",
                "last_name" : "kumar"
            },
            {
                "watchlistid" : "WL128",
                "first_name" : "ajit",
                "last_name" : "kumar"
            }
        ],  #watchlistdata  #all_details

        "person_dict_for_matching": {
            'lastname': 'Al-Fayez',
            'firstname': 'Waleed',
            'middlename': 'Abdullah',
            'aliasname': 'Waleed Abdullah Al-Fayez',
            'dateofbirth': 'NaT',
            'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
            'residence': 'SA',
            'citizenship': 'SA',
            'address_street': '23 Al-Masjid St, Riyadh',
            'entity': 'Entity'
        }, #input

        "full_records": [
            {
                "watchlistid" : "WL123",
                'lastname': 'Al-Fayez',
                'firstname': 'Waleed',
                'middlename': 'Abdullah',
                'aliasname': 'Waleed Abdullah Al-Fayez',
                'dateofbirth': 'NaT',
                'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
                'residence': 'SA',
                'citizenship': 'SA',
                'address_street': '23 Al-Masjid St, Riyadh',
                'entity': 'Entity'
            },
            {
                "watchlistid" : "WL124",
                'lastname': 'ajay',
                'firstname': 'singh',
                'middlename': 'Abdullah',
                'aliasname': 'Waleed Abdullah Al-Fayez',
                'dateofbirth': 'NaT',
                'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
                'residence': 'SA',
                'citizenship': 'SA',
                'address_street': '23 Al-Masjid St, Riyadh',
                'entity': 'Entity'
            },
            {
                "watchlistid" : "WL125",
                'lastname': 'Al-Fayez',
                'firstname': 'Waleed',
                'middlename': 'Abdullah',
                'aliasname': 'Waleed Abdullah Al-Fayez',
                'dateofbirth': '',
                'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
                'residence': 'bangladesh',
                'citizenship': 'bangladesh',
                'address_street': '',
                'entity': 'Entity'
            },
            {
                "watchlistid" : "WL126",
                "first_name" : "vinay",
                "last_name" : "kumar",
                'middlename': 'Abdullah',
                'aliasname': 'Waleed Abdullah Al-Fayez',
                'dateofbirth': '',
                'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
                'residence': 'bangladesh',
                'citizenship': 'bangladesh',
                'address_street': '',
                'entity': 'Entity'
            },
            {
                "watchlistid" : "WL127",
                "first_name" : "vikky",
                "last_name" : "kumar",
                'middlename': 'Abdullah',
                'aliasname': 'Waleed Abdullah Al-Fayez',
                'dateofbirth': '',
                'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
                'residence': 'bangladesh',
                'citizenship': 'bangladesh',
                'address_street': '',
                'entity': 'Entity'
            },
            {
                "watchlistid" : "WL128",
                "first_name" : "ajit",
                "last_name" : "verma",
                'middlename': 'Abdullah',
                'aliasname': 'Waleed Abdullah Al-Fayez',
                'dateofbirth': '',
                'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
                'residence': 'bangladesh',
                'citizenship': 'bangladesh',
                'address_street': '',
                'entity': 'Entity'
            }
        ], #all_details

        "customer_details_for_closing": {
            'lastname': 'Al-Fayez',
            'firstname': 'Waleed',
            'middlename': 'Abdullah',
            'aliasname': 'Waleed Abdullah Al-Fayez',
            'dateofbirth': 'NaT',
            'date_of_incorporation': 'Timestamp(1992-01-05 00:00:00)',
            'residence': 'SA',
            'citizenship': 'SA',
            'address_street': '23 Al-Masjid St, Riyadh',
            'entity': 'Entity'
        } #input
    }

    final_state = graph.invoke(initial_state)

    print(final_state.get("closing_output", []))

if __name__ == "__main__":
    main()
