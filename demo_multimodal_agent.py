"""Demo: Multimodal LangGraph Agent with BigQuery Observability.

This script creates a multimodal LangGraph agent that processes images
and logs all events (including multimodal content) to BigQuery.

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community[bigquery] langgraph langchain-google-genai pillow
"""

import asyncio
import base64
import io
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")
TABLE_ID = "agent_events_v2"


class AgentState(TypedDict):
    """State for the multimodal agent."""

    messages: Annotated[list[BaseMessage], add_messages]


# --- Tools ---
@tool
def lookup_landmark(name: str) -> str:
    """Look up information about a landmark or location.

    Args:
        name: The name of the landmark or location.

    Returns:
        Information about the landmark.
    """
    landmarks = {
        "eiffel tower": (
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
            "Built 1887-1889. Height: 330m. Visited by ~7 million people annually."
        ),
        "golden gate bridge": (
            "The Golden Gate Bridge is a suspension bridge in San Francisco, CA. "
            "Opened 1937. Span: 1,280m. Color: International Orange."
        ),
        "colosseum": (
            "The Colosseum is an ancient amphitheatre in Rome, Italy. "
            "Built 72-80 AD. Capacity: 50,000-80,000 spectators."
        ),
        "statue of liberty": (
            "The Statue of Liberty is a neoclassical sculpture on Liberty Island, "
            "New York. Dedicated 1886. Height: 93m (with pedestal)."
        ),
        "big ben": (
            "Big Ben is the nickname for the Great Bell in the Elizabeth Tower at "
            "the Palace of Westminster, London. Completed 1859."
        ),
        "sydney opera house": (
            "The Sydney Opera House is a multi-venue performing arts centre in "
            "Sydney, Australia. Opened 1973. Designed by Jørn Utzon."
        ),
    }
    name_lower = name.lower().strip()
    for key, info in landmarks.items():
        if key in name_lower or name_lower in key:
            return info
    return f"No information found for '{name}'. Try: Eiffel Tower, Golden Gate Bridge, Colosseum."


@tool
def get_travel_tips(destination: str) -> str:
    """Get travel tips for a destination.

    Args:
        destination: The travel destination city or country.

    Returns:
        Travel tips for the destination.
    """
    tips = {
        "paris": (
            "Paris Travel Tips:\n"
            "- Best time to visit: April-June or September-October\n"
            "- Get a Paris Museum Pass for skip-the-line access\n"
            "- The Metro is the fastest way to get around\n"
            "- Tip: Visit the Eiffel Tower at sunset for the best views"
        ),
        "france": (
            "France Travel Tips:\n"
            "- TGV high-speed trains connect major cities\n"
            "- Shops often close 12-2pm for lunch\n"
            "- Tipping is included in restaurant bills (service compris)"
        ),
        "rome": (
            "Rome Travel Tips:\n"
            "- Book Colosseum tickets in advance to skip lines\n"
            "- Dress modestly for Vatican/church visits\n"
            "- Best gelato is found away from tourist areas"
        ),
        "italy": (
            "Italy Travel Tips:\n"
            "- Regional trains are affordable and scenic\n"
            "- Coffee culture: espresso at the bar is cheapest\n"
            "- Many museums close on Mondays"
        ),
    }
    dest_lower = destination.lower().strip()
    for key, info in tips.items():
        if key in dest_lower or dest_lower in key:
            return info
    return f"No specific tips for '{destination}'. General tip: research local customs before traveling!"


def _create_test_image() -> str:
    """Create a simple test image and return as base64 data URL.

    Creates a 200x200 pixel image with a simple scene.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("pip install Pillow")

    img = Image.new("RGB", (200, 150), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)

    # Draw a simple Eiffel Tower silhouette
    # Base
    draw.polygon([(70, 140), (130, 140), (115, 60), (85, 60)], fill=(100, 100, 100))
    # Top
    draw.polygon([(90, 60), (110, 60), (105, 20), (95, 20)], fill=(80, 80, 80))
    # Antenna
    draw.line([(100, 20), (100, 5)], fill=(60, 60, 60), width=2)
    # Ground
    draw.rectangle([(0, 140), (200, 150)], fill=(34, 139, 34))
    # Text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((55, 142), "Paris, France", fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_multimodal_agent() -> StateGraph:
    """Create a multimodal LangGraph agent.

    Returns:
        Compiled LangGraph agent.
    """
    tools = [lookup_landmark, get_travel_tips]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=PROJECT_ID,
    ).bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        """Async agent node."""
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


async def main() -> None:
    """Run the multimodal agent demo."""
    print("=" * 60)
    print("Multimodal Agent with BigQuery Observability")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}.{TABLE_ID}")

    # 1. Configure handler with multimodal content logging
    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
        log_multi_modal_content=True,
    )

    handler = AsyncBigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        config=config,
        graph_name="multimodal_travel_agent",
    )

    agent = create_multimodal_agent()

    # 2. Create a test image (Eiffel Tower sketch)
    print("\nCreating test image...")
    image_data_url = _create_test_image()
    print(f"Image size: {len(image_data_url)} chars (base64)")

    # 3. Run: Image analysis with tool calls
    print("\n" + "=" * 60)
    print("Test: Multimodal Image Analysis")
    print("=" * 60)

    metadata = {
        "session_id": "multimodal-session-001",
        "user_id": "demo-user",
        "agent": "multimodal_travel_agent",
    }

    multimodal_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "What landmark is shown in this image? "
                    "Look up information about it and give me travel tips "
                    "for visiting."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            },
        ]
    )

    async with handler.graph_context("multimodal_travel_agent", metadata=metadata):
        result = await agent.ainvoke(
            {"messages": [multimodal_message]},
            config={
                "callbacks": [handler],
                "metadata": metadata,
            },
        )

    final_message = result["messages"][-1]
    response = (
        final_message.content
        if isinstance(final_message, AIMessage)
        else str(final_message)
    )
    print(f"Response:\n{response}")

    # 4. Shut down
    print("\n" + "=" * 60)
    print("Shutting down handler...")
    await handler.shutdown()
    print("Events flushed to BigQuery successfully.")

    print(f"""
Done! Query multimodal events:

-- Multimodal event trace
SELECT
    timestamp,
    event_type,
    JSON_VALUE(attributes, '$.tool_name') AS tool_name,
    JSON_VALUE(latency_ms, '$.total_ms') AS latency_ms,
    ARRAY_LENGTH(content_parts) AS num_content_parts,
    (SELECT cp.mime_type FROM UNNEST(content_parts) cp WHERE cp.mime_type != 'text/plain' LIMIT 1) AS media_type,
    status
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
WHERE agent = 'multimodal_travel_agent'
  AND DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 20;

-- Content parts detail
SELECT
    timestamp,
    event_type,
    cp.part_index,
    cp.mime_type,
    cp.storage_mode,
    LEFT(cp.text, 100) AS text_preview
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`,
  UNNEST(content_parts) AS cp
WHERE agent = 'multimodal_travel_agent'
  AND DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp, cp.part_index;
    """)


if __name__ == "__main__":
    asyncio.run(main())
