"""Demo: Travel Assistant Agent with BigQuery Observability.

This script creates a LangGraph agent with 3 tools (time, weather, currency)
and logs every event to BigQuery via the BigQueryCallbackHandler.
"""

import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)
from langchain_google_genai import ChatGoogleGenerativeAI


# --- Tools ---
@tool
def get_current_time() -> str:
    """Returns the current local time."""
    now = datetime.now()
    return (
        f"Current time: {now.strftime('%I:%M:%S %p')} "
        f"on {now.strftime('%B %d, %Y')}"
    )


@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a specific city."""
    weather_data = {
        "new york": {"temp": 22, "condition": "Clear"},
        "tokyo": {"temp": 24, "condition": "Sunny"},
        "london": {"temp": 14, "condition": "Overcast"},
        "paris": {"temp": 18, "condition": "Partly Cloudy"},
        "sydney": {"temp": 28, "condition": "Warm and Clear"},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return (
            f"Weather in {city.title()}: "
            f"{data['temp']}°C, {data['condition']}"
        )
    return f"Weather data for '{city}' not available."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies."""
    rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}
    from_curr = from_currency.upper()
    to_curr = to_currency.upper()
    if from_curr not in rates or to_curr not in rates:
        return "Unknown currency"
    result = amount * rates[from_curr] / rates[to_curr]
    return f"{amount} {from_curr} = {result:.2f} {to_curr}"


def main() -> None:
    project_id = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
    dataset_id = "agent_analytics"
    table_id = "agent_events_v2"

    print(f"Project: {project_id}")
    print(f"Dataset: {dataset_id}.{table_id}")
    print()

    # 1. Configure the callback handler
    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
    )

    handler = BigQueryCallbackHandler(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        config=config,
        graph_name="travel_assistant",
    )

    # 2. Create the LLM and agent
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=project_id,
    )
    tools = [get_current_time, get_weather, convert_currency]
    agent = create_agent(llm, tools)

    # 3. Run the agent with graph_context
    run_metadata = {
        "session_id": "demo-session-001",
        "user_id": "demo-user",
        "agent": "travel_assistant",
    }

    query = (
        "What time is it right now? "
        "What's the weather like in Tokyo? "
        "And how much is 100 USD in EUR?"
    )

    print(f"Query: {query}")
    print()
    print("Running agent...")
    print("-" * 60)

    with handler.graph_context("travel_assistant", metadata=run_metadata):
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [handler],
                "metadata": run_metadata,
            },
        )

    # 4. Print the response
    response = result["messages"][-1].content
    print(f"\nAgent Response:\n{response}")
    print("-" * 60)

    # 5. Clean up
    handler.shutdown()
    print("\nEvents flushed to BigQuery successfully.")
    print(
        f"\nQuery your events:\n"
        f"  bq query --use_legacy_sql=false \\\n"
        f"    'SELECT timestamp, event_type, "
        f"JSON_VALUE(latency_ms, \"$.total_ms\") as latency_ms "
        f"FROM `{project_id}.{dataset_id}.{table_id}` "
        f"WHERE session_id = \"demo-session-001\" "
        f"ORDER BY timestamp'"
    )


if __name__ == "__main__":
    main()
